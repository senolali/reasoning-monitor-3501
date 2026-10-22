import hashlib
import json
import runpy
import socket
from pathlib import Path

import pytest

from reasoning_monitor.lost_conversation import MathTask
from reasoning_monitor.lost_conversation_audit import (
    exact_paired_power,
    exposure_audit,
    inside_root,
    measurement_budget,
    normalized_question,
    ranked_sample,
    run_audit,
    sample_plan,
    semantic_audit,
    write_artifacts,
)


@pytest.fixture
def tasks():
    return [MathTask(f"sharded-GSM8K/{i}", f"Pat has {i + 10} red balls. Total?", "22",
                     ("How many?", "There are boxes.", "More boxes.", "Each contains balls.",
                      "The final count.")) for i in range(6)]


@pytest.fixture
def review(tasks):
    return {"source_sha256": "sha", "reviewer": "assistant_first_pass",
            "independent_review_complete": False,
            "tasks": [{"task_id": t.task_id, "total_turns": len(t.shards),
                       "earliest_sufficient_turn_candidate": 5, "status": "candidate",
                       "reason": "Last count needed"} for t in tasks]}


@pytest.fixture
def config():
    return dict(protocol_id="LIC-RQ-v1", stage="call_free_design",
                execution_authorized=False, revelation_order="source_shard_id",
                termination="all_shards_no_correctness_stop", primary_response_index=0,
                conditions=["dialogue_history", "fresh_prefix"],
                models=["gpt4o_mini", "llama4_scout"], k_repeats=5,
                l_paraphrases=2, max_output_tokens=2048, es_window=5)


def historical_file(root, rows):
    path = root / "results/completed/data/records_test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_exposure_uses_prompt_not_outcome_or_qid(tmp_path, tasks):
    source = historical_file(tmp_path, [
        {"dataset": "gsm8k", "qid": "unrelated-local-id", "prompt": tasks[0].question + "\nOutput format:",
         "raw_text": tasks[1].question, "gold_answer": tasks[2].question},
        {"dataset": "gsm8k", "qid": tasks[1].task_id, "prompt": "A different question"},
        {"dataset": "math", "prompt": tasks[3].question},
    ])
    report = exposure_audit(tmp_path, tasks, ["results/completed"])
    assert report["status_counts"] == {"exact_question_overlap": 1, "no_match_in_allowlist": 5}
    assert report["source_files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["tasks"][0]["exact_evidence"][0]["line"] == 1


def test_whitespace_case_but_not_numbers_normalized(tmp_path, tasks):
    historical_file(tmp_path, [{"dataset": "gsm8k", "prompt": tasks[0].question.upper().replace(" ", "  ")}])
    report = exposure_audit(tmp_path, tasks, ["results/completed"])
    assert report["status_counts"]["exact_question_overlap"] == 1
    assert normalized_question("-1.5") != normalized_question("1.5")
    assert normalized_question("1.5") != normalized_question("15")


def test_encoding_variant_only_candidate(tmp_path):
    task = MathTask("t", "Alex\ufffds total is 10. Why?", "10", ("a", "b"))
    historical_file(tmp_path, [{"dataset": "gsm8k", "prompt": "Alex's total is 10. Why?"}])
    report = exposure_audit(tmp_path, [task], ["results/completed"])
    assert report["tasks"][0]["status"] == "encoding_variant_candidate"
    assert not report["tasks"][0]["exact_evidence"]


def test_closed_files_not_opened(tmp_path, tasks):
    historical_file(tmp_path, [{"dataset": "gsm8k", "prompt": "different"}])
    closed = tmp_path / "results/closed/data"
    closed.mkdir(parents=True)
    (closed / "records_holdout.jsonl").write_text("intentionally invalid JSON")
    assert len(exposure_audit(tmp_path, tasks, ["results/completed"])["source_files"]) == 1


@pytest.mark.parametrize("campaigns", [[], ["results/missing"], ["../outside"], ["a", "a"]])
def test_bad_or_missing_exposure_source_fails_closed(tmp_path, tasks, campaigns):
    with pytest.raises(ValueError):
        exposure_audit(tmp_path, tasks, campaigns)


def test_missing_input_is_not_silently_unseen(tmp_path, tasks):
    historical_file(tmp_path, [{"dataset": "gsm8k", "prompt": ""}])
    with pytest.raises(ValueError, match="Missing GSM8K"):
        exposure_audit(tmp_path, tasks, ["results/completed"])


def test_semantic_candidates_do_not_authorize_scoring(tasks, review):
    review["tasks"][0]["earliest_sufficient_turn_candidate"] = 3
    audit = semantic_audit(tasks, review, "sha")
    assert audit["candidate_early_sufficient_tasks"] == 1
    assert audit["candidate_sufficient_partial_prefixes"] == 2
    assert not any(p["execution_eligibility_authorized"] for p in audit["prefixes"])
    assert sum(p["baseline_full_information_eligible"] for p in audit["prefixes"]) == 6
    assert not audit["independent_review_complete"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "turn", "count", "reviewer", "hash", "quarantine"])
def test_invalid_semantic_annotations_rejected(tasks, review, change):
    if change == "missing":
        review["tasks"].pop()
    elif change == "duplicate":
        review["tasks"].append(review["tasks"][0])
    elif change == "turn":
        review["tasks"][0]["earliest_sufficient_turn_candidate"] = True
    elif change == "count":
        review["tasks"][0]["total_turns"] = 3
    elif change == "reviewer":
        review["independent_review_complete"] = True
    elif change == "hash":
        review["source_sha256"] = "changed"
    else:
        review["tasks"][0]["status"] = "quarantine"
    with pytest.raises(ValueError):
        semantic_audit(tasks, review, "sha")


def test_whole_task_sampling_order_independent(tasks):
    assert ranked_sample(tasks, 3, "seed") == ranked_sample(tasks[::-1], 3, "seed")
    with pytest.raises(ValueError):
        ranked_sample(tasks, 100, "seed")
    with pytest.raises(ValueError, match="Duplicate"):
        ranked_sample(tasks + tasks, 2, "seed")


def test_sample_excludes_overlap_candidate_and_quarantine(tasks, review, config):
    review["tasks"][2].update(status="quarantine", earliest_sufficient_turn_candidate=None)
    semantics = semantic_audit(tasks, review, "sha")
    exposure = {"tasks": [{"task_id": t.task_id, "status": status} for t, status in zip(
        tasks, ["exact_question_overlap", "encoding_variant_candidate"] + ["no_match_in_allowlist"] * 4, strict=True)]}
    plan = sample_plan(tasks, exposure, semantics, config,
                       {"sampling_seed": "x", "integration_tasks": 1, "science_task_options": [2, 10]})
    assert plan["eligible_unmatched_candidate_tasks"] == 3
    assert not plan["options"][1]["available"]
    science = plan["options"][0]["science_task_ids"]
    integration = plan["integration_task_ids"]
    assert set(science).isdisjoint(integration)
    assert set(science + integration) == {t.task_id for t in tasks[3:]}
    assert not plan["execution_authorized"]


def test_budget_warmup_and_primary_included_once(tasks, review, config):
    review["tasks"][0]["earliest_sufficient_turn_candidate"] = 3
    semantics = semantic_audit(tasks, review, "sha")
    budget = measurement_budget(tasks, config, semantics)
    assert budget["logical_model_calls"] == 6 * 5 * 2 * 2 * 7
    assert budget["candidate_gold_sufficient_prefixes_upper_bound"] == 8
    assert budget["candidate_RQ6_prefixes_upper_bound_after_ES_warmup"] == 6
    assert budget["candidate_RQ6_coverage_upper_bound"] == 0.2
    semantics["tasks"][0]["status"] = "quarantine"
    with pytest.raises(ValueError, match="Quarantined"):
        measurement_budget(tasks, config, semantics)


def test_paired_power_known_limits_and_external_reference():
    from scipy.stats import binom, binomtest

    assert exact_paired_power(1, 1, 0) == 0
    assert exact_paired_power(6, 1, 0) == 1
    assert exact_paired_power(20, 0, 0) == 0
    n, b, c = 10, 0.3, 0.1
    reference = sum(binom.pmf(d, n, b + c) * binom.pmf(k, d, b / (b + c))
                    for d in range(1, n + 1) for k in range(d + 1)
                    if binomtest(k, d, 0.5).pvalue <= 0.05)
    assert exact_paired_power(n, b, c) == pytest.approx(reference)
    assert exact_paired_power(48, 0.25, 0.05) == pytest.approx(0.6754150632972747)
    with pytest.raises(ValueError):
        exact_paired_power(12, float("nan"), 0.1)


def test_artifacts_idempotent_no_partial_replacement(tmp_path):
    artifacts = {"report.json": {"test": 1}, "plan.json": {"x": 2}}
    write_artifacts(tmp_path, artifacts)
    write_artifacts(tmp_path, artifacts)
    (tmp_path / "plan.json").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing artifact differs"):
        write_artifacts(tmp_path, {"new.json": {"z": 0}, **artifacts})
    assert (tmp_path / "plan.json").read_text() == "keep"
    assert not (tmp_path / "new.json").exists()


def test_cli_no_network_no_write_by_default(tmp_path, monkeypatch, config):
    import yaml

    rows = [{"task": "math", "task_id": f"t{i}", "question": f"Question {i}?", "answer": "#### 1",
             "shards": [{"shard_id": 1, "shard": "first"}, {"shard_id": 2, "shard": "last"}]}
            for i in range(3)]
    source = json.dumps(rows).encode()
    (tmp_path / "source.json").write_bytes(source)
    sha = hashlib.sha256(source).hexdigest()
    config["source"] = {"path": "source.json", "sha256": sha}
    (tmp_path / "design.yaml").write_text(yaml.safe_dump(config))
    review = {"source_sha256": sha, "reviewer": "assistant_first_pass", "independent_review_complete": False,
              "tasks": [{"task_id": r["task_id"], "total_turns": 2,
                         "earliest_sufficient_turn_candidate": 2, "status": "candidate", "reason": "last"}
                        for r in rows]}
    (tmp_path / "review.json").write_text(json.dumps(review))
    historical_file(tmp_path, [{"dataset": "gsm8k", "prompt": "no overlap"}])
    settings = dict(execution_authorized=False, design_path="design.yaml", semantic_review_path="review.json",
                    completed_campaigns=["results/completed"], sampling_seed="seed", integration_tasks=1,
                    science_task_options=[2], output_dir="out")
    (tmp_path / "audit.yaml").write_text(yaml.safe_dump(settings))

    def deny(*args, **kwargs):
        raise AssertionError("No network during input audit")

    monkeypatch.setattr(socket, "create_connection", deny)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_lost_conversation.py"
    main = runpy.run_path(str(script))["main"]
    main.__globals__["ROOT"] = tmp_path
    assert main(["--config", "audit.yaml"]) == 0
    assert not (tmp_path / "out").exists()
    assert main(["--config", "audit.yaml", "--write"]) == 0
    assert main(["--config", "audit.yaml", "--write"]) == 0
    assert json.loads((tmp_path / "out/assessment.json").read_text())["no_model_calls"]
