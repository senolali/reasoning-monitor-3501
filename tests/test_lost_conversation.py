import hashlib
import json
import runpy
from pathlib import Path

import pytest

from reasoning_monitor.lost_conversation import (
    METRICS,
    MathTask,
    build_prefix_messages,
    load_math_tasks,
    observability_mask,
    prefix_metadata,
    preparation_report,
    six_metric_summary,
    validate_design,
)


@pytest.fixture
def task():
    return MathTask("sharded-GSM8K/example", "HIDDEN FULL QUESTION", "777777",
                    ("What is the total?", "There are three teams.",
                     "Each has FUTURE_SECRET members."))


@pytest.fixture
def config():
    return dict(protocol_id="LIC-RQ-v1", stage="call_free_design",
                execution_authorized=False, revelation_order="source_shard_id",
                termination="all_shards_no_correctness_stop", primary_response_index=0,
                conditions=["dialogue_history", "fresh_prefix"],
                models=["gpt4o_mini", "llama4_scout"], k_repeats=5,
                l_paraphrases=2, max_output_tokens=2048, es_window=5)


def fixture_source(tmp_path, **changes):
    row = dict(task="math", task_id="sharded-GSM8K/1", question="A question",
               answer="calculation\n#### 1,234",
               shards=[{"shard_id": 1, "shard": "Question?"},
                       {"shard_id": 2, "shard": "The last fact."}])
    row.update(changes)
    raw = json.dumps([row]).encode()
    path = tmp_path / "source.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_real_shard_key_and_reference_parsing(tmp_path):
    tasks, counts = load_math_tasks(*fixture_source(tmp_path))
    assert tasks[0].gold_answer == "1,234"
    assert tasks[0].shards == ("Question?", "The last fact.")
    assert counts == {"math": 1}


def test_hash_mismatch_fails_closed(tmp_path):
    path, _ = fixture_source(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_math_tasks(path, "0" * 64)


@pytest.mark.parametrize("changes", [
    {"shards": [{"shard_id": 1, "shard": "a"}, {"shard_id": 3, "shard": "b"}]},
    {"shards": [{"shard_id": True, "shard": "a"}, {"shard_id": 2, "shard": "b"}]},
    {"answer": "no final delimiter"}, {"answer": "#### NaN"},
    {"answer": "#### 2 #### 3"}, {"answer": "#### not a number"},
])
def test_malformed_math_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load_math_tasks(*fixture_source(tmp_path, **changes))


def test_duplicate_question_rejected(tmp_path):
    path, _ = fixture_source(tmp_path)
    rows = json.loads(path.read_text())
    rows.append(dict(rows[0], task_id="different", question=" a   QUESTION "))
    raw = json.dumps(rows).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="Duplicate"):
        load_math_tasks(path, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("condition", ["dialogue_history", "fresh_prefix"])
def test_no_future_full_question_or_gold(task, condition):
    history = ["Please give the team count."] if condition == "dialogue_history" else []
    messages = build_prefix_messages(task, 2, condition, history)
    text = json.dumps(messages)
    assert "three teams" in text
    for secret in (task.question, task.gold_answer, "FUTURE_SECRET"):
        assert secret not in text
    changed = MathTask(task.task_id, "another full question", "99999",
                       task.shards[:2] + ("entirely changed future",))
    assert messages == build_prefix_messages(changed, 2, condition, history)


def test_first_turn_identical(task):
    assert build_prefix_messages(task, 1, "dialogue_history") == build_prefix_messages(
        task, 1, "fresh_prefix")


def test_history_required_and_control_clean(task):
    with pytest.raises(ValueError, match="Exactly one"):
        build_prefix_messages(task, 2, "dialogue_history")
    with pytest.raises(ValueError, match="Exactly one"):
        build_prefix_messages(task, 2, "dialogue_history", ["primary", "repeat"])
    with pytest.raises(ValueError, match="cannot contain"):
        build_prefix_messages(task, 2, "fresh_prefix", ["primary"])
    assert "primary only" in json.dumps(
        build_prefix_messages(task, 2, "dialogue_history", ["primary only"]))


@pytest.mark.parametrize("turn", [0, -1, 4, True, 1.5])
def test_invalid_turn(task, turn):
    with pytest.raises(ValueError):
        prefix_metadata(task, turn)


def test_partial_answerability_unknown_not_wrong(task):
    early = prefix_metadata(task, 1)
    assert early["answerability"] == "unreviewed_prefix"
    assert not early["gold_scoring_eligible"]
    assert "gold_answer" not in early
    assert prefix_metadata(task, 3)["gold_scoring_eligible"]
    values = dict.fromkeys(METRICS, 1.0)
    statuses = dict.fromkeys(METRICS, "measured")
    masked, reasons = observability_mask(
        values, statuses, all_information_available=False, response_kind="answer_attempt")
    for metric in ("CQ", "RS", "ES"):
        assert masked[metric] is None
        assert reasons[metric] == "prefix_answerability_unreviewed"
    assert masked["CS"] == 1.0
    assert values["CQ"] == 1.0


def test_clarification_not_wrong():
    values, statuses = observability_mask(
        dict.fromkeys(METRICS, 0.0), dict.fromkeys(METRICS, "measured"),
        all_information_available=True, response_kind="clarification")
    assert values["CQ"] is None
    assert statuses["CQ"] == "no_classified_answer_attempt"


def test_composite_requires_six_measured_values():
    values, statuses = dict.fromkeys(METRICS, 1.0), dict.fromkeys(METRICS, "measured")
    assert six_metric_summary(values, statuses)["RQ6_mean"] == 1.0
    values["RS"] = None
    statuses["RS"] = "C_subset_excluded"
    summary = six_metric_summary(values, statuses)
    assert summary["RQ6_mean"] is None
    assert summary["measured_metric_count"] == 5
    assert not summary["online_warning_feature"]


@pytest.mark.parametrize("status,value", [("warmup", 0.9), ("measured", float("nan")),
                                         ("measured", None)])
def test_warmup_or_nonfinite_prevents_composite(status, value):
    values, statuses = dict.fromkeys(METRICS, 1.0), dict.fromkeys(METRICS, "measured")
    values["ES"], statuses["ES"] = value, status
    assert six_metric_summary(values, statuses)["RQ6_mean"] is None


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5"])
def test_invalid_score_not_clipped(value):
    values = dict.fromkeys(METRICS, 1.0)
    values["CS"] = value
    with pytest.raises(ValueError):
        six_metric_summary(values, dict.fromkeys(METRICS, "measured"))


def test_budget_counts_primary_once(task, config):
    report = preparation_report([task], {"math": 1}, config)
    assert report["target_model_logical_calls"] == 2 * 2 * 3 * 7
    assert report["full_information_prefixes"] == 1
    assert report["unreviewed_partial_prefixes"] == 2
    assert not report["execution_ready"]
    assert not report["execution_authorized"]
    config["execution_authorized"] = True
    with pytest.raises(ValueError):
        validate_design(config)


def test_cli_offline_idempotent_no_overwrite(tmp_path, monkeypatch, config):
    import socket

    import yaml

    script = Path(__file__).resolve().parents[1] / "scripts/prepare_lost_conversation.py"
    main = runpy.run_path(str(script))["main"]
    main.__globals__["ROOT"] = tmp_path
    source, sha = fixture_source(tmp_path)
    config.update(source={"path": source.name, "sha256": sha}, output_dir="out")
    (tmp_path / "design.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    def deny_network(*args, **kwargs):
        raise AssertionError("Network forbidden during preparation")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    assert main(["--config", "design.yaml"]) == 0
    assert not (tmp_path / "out").exists()
    assert main(["--config", "design.yaml", "--write"]) == 0
    assert main(["--config", "design.yaml", "--write"]) == 0
    report = tmp_path / "out/assessment.json"
    report.write_text("preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing artifact differs"):
        main(["--config", "design.yaml", "--write"])
    assert report.read_text() == "preserve me"


def test_path_cannot_escape_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_lost_conversation.py"
    resolve = runpy.run_path(str(script))["project_path"]
    resolve.__globals__["ROOT"] = tmp_path
    with pytest.raises(ValueError, match="inside this repository"):
        resolve("../elsewhere")
