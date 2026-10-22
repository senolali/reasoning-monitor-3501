"""Benchmark loading.

Datasets are the ones the reasoning-evaluation literature has converged on:

  GSM8K          Cobbe et al. 2021   grade-school math word problems
  ARC-Challenge  Clark et al. 2018   science exam questions
  StrategyQA     Geva et al. 2021    implicit multi-hop yes/no reasoning
  MATH            Hendrycks et al. 2021 competition mathematics

They download from the HuggingFace Hub on first use and are cached under
`data/raw/`. If the hub is unreachable — offline CI, an air-gapped machine, a
reviewer reproducing without network — a deterministic synthetic corpus with
the same schema is generated instead, so `make repro` never fails for network
reasons. Which source was used is recorded in the run manifest, because results
from the synthetic corpus are NOT comparable to results from the real one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from reasoning_monitor.types import Query

log = logging.getLogger(__name__)

HF_SPECS = {
    "gsm8k": dict(path="openai/gsm8k", name="main", split="test",
                  question="question", answer="answer", domain="math"),
    "e2h_gsm8k": dict(
        path="furonghuang-lab/Easy2Hard-Bench",
        name="E2H-GSM8K",
        split="eval",
        revision="55bc0d2fb10954151e669d2026b87fa896f2fa26",
        question="question",
        answer="answer",
        domain="math",
    ),
    "e2h_amc": dict(
        path="furonghuang-lab/Easy2Hard-Bench",
        name="E2H-AMC",
        split="eval",
        revision="55bc0d2fb10954151e669d2026b87fa896f2fa26",
        question="problem",
        answer="answer",
        domain="math_competition",
    ),
    "arc_challenge": dict(path="allenai/ai2_arc", name="ARC-Challenge", split="test",
                          question="question", answer="answerKey", domain="science"),
    "strategyqa": dict(path="ChilleD/StrategyQA", name=None, split="test",
                       question="question", answer="answer", domain="commonsense"),
    # MATH (Hendrycks et al. 2021) — competition-level problems, domain math_hard.
    # gold answer is the last boxed expression in the solution string.
    "math": dict(
        path="EleutherAI/hendrycks_math",
        name=(
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ),
        split="test",
        question="problem",
        answer="solution",
        domain="math_hard",
    ),
    "supergpqa": dict(
        path="m-a-p/SuperGPQA",
        split="train",
        revision="4430d4458112c7d4497fdcf94d7cc223313d6acf",
        source_file="SuperGPQA-all.jsonl",
        cache_folder="m-a-p___SuperGPQA",
        domain="analytical_chemistry",
    ),
}


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    queries: list[Query]
    source: str          # "huggingface" or "synthetic"
    content_hash: str


def _extract_gold(dataset_name: str, raw: str) -> str:
    """Pull the final answer out of a dataset's answer field."""
    if dataset_name in {"gsm8k", "e2h_gsm8k"}:
        # GSM8K answers end with "#### <number>".
        return raw.split("####")[-1].strip().replace(",", "")
    if dataset_name == "math":
        # MATH solutions may contain nested braces and more than one boxed
        # expression.  Use the last balanced \boxed{...}, which conventionally
        # carries the final answer.
        boxed = _last_boxed_expression(raw)
        if boxed is not None:
            return boxed
        # Fallback: last numeric token in the solution.
        import re
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
        return nums[-1] if nums else raw.strip()
    return str(raw).strip()


def _last_boxed_expression(text: str) -> str | None:
    marker = "\\boxed{"
    starts = [i for i in range(len(text)) if text.startswith(marker, i)]
    for start in reversed(starts):
        depth = 1
        content_start = start + len(marker)
        for pos in range(content_start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:pos].strip()
    return None


def _arc_prompt_and_gold(row: dict) -> tuple[str, str]:
    """Render ARC-Challenge as an answerable labelled multiple-choice item."""
    question = str(row.get("question", "")).strip()
    choices = row.get("choices") or {}
    labels = [str(value).strip() for value in choices.get("label", [])]
    texts = [str(value).strip() for value in choices.get("text", [])]
    if not labels or len(labels) != len(texts):
        raise ValueError("ARC-Challenge row has missing or malformed choices")

    raw_gold = str(row.get("answerKey", "")).strip()
    if raw_gold in labels:
        gold = raw_gold
    elif raw_gold.isdigit() and 1 <= int(raw_gold) <= len(labels):
        gold = labels[int(raw_gold) - 1]
    else:
        gold = raw_gold
    rendered = "\n".join(
        [
            question,
            "Choices:",
            *[
                f"{label}. {text}"
                for label, text in zip(labels, texts, strict=True)
            ],
        ]
    )
    return rendered, gold


def _solution_step_count(dataset_name: str, raw: str) -> int | None:
    """Return a dataset-backed reasoning-step proxy without parsing model text."""
    if dataset_name not in {"gsm8k", "e2h_gsm8k"}:
        return None
    import re
    operations = re.findall(r"<<[^<>]+>>", raw)
    return len(operations) if operations else None


def _hash_queries(queries: list[Query]) -> str:
    h = hashlib.sha256()
    for q in queries:
        h.update(q.qid.encode())
        h.update(q.prompt.encode())
        h.update(q.gold_answer.encode())
        h.update(str(q.difficulty_rating).encode())
        h.update(str(q.difficulty_rating_std).encode())
        h.update(str(q.difficulty_quantile).encode())
        h.update(str(q.difficulty_model_avg_acc).encode())
    return h.hexdigest()[:16]


def _render_supergpqa_prompt(row: dict, prompt_mode: str = "choice_last") -> str:
    options = list(row.get("options") or ())
    labels = [chr(ord("A") + index) for index in range(len(options))]
    instructions = {
        "choice_last": [
            "Give concise reasoning, then end with exactly: FINAL_CHOICE: X"
        ],
        "choice_first": [
            "Your first output line must be exactly: FINAL_CHOICE: X",
            "After that line, give at most four short sentences of reasoning and stop.",
        ],
    }
    if prompt_mode not in instructions:
        raise ValueError(
            f"unknown SuperGPQA prompt mode {prompt_mode!r}; "
            f"expected one of {list(instructions)}"
        )
    return "\n".join(
        [
            str(row.get("question", "")).strip(),
            "Options:",
            *[
                f"{label}. {str(option).strip()}"
                for label, option in zip(labels, options, strict=True)
            ],
            *instructions[prompt_mode],
        ]
    )


def _render_answer_first_prompt(query: Query) -> str:
    """Append a typed first-line answer contract without changing task content."""
    if query.dataset == "arc_challenge":
        answer_contract = [
            "Your first output line must be exactly: FINAL_CHOICE: X",
            "Replace X with exactly one option letter from A through D.",
        ]
    elif query.dataset in {"gsm8k", "math"}:
        answer_contract = [
            "Your first output line must have exactly this form: FINAL_ANSWER: <answer>",
            "Keep the complete final answer on that single line.",
        ]
    else:
        raise ValueError(
            "answer_first prompt mode supports only gsm8k, arc_challenge and math; "
            f"received {query.dataset!r}"
        )
    return "\n".join(
        [
            query.prompt.rstrip(),
            "Output format:",
            *answer_contract,
            "After the first line, give at most eight short numbered reasoning steps and stop.",
            "Never revise the first-line answer.",
        ]
    )


def _render_answer_last_bounded_prompt(query: Query) -> str:
    """Require a short solution followed by one typed final-answer line."""
    if query.dataset == "arc_challenge":
        answer_contract = [
            "Your final output line must be exactly: FINAL_CHOICE: X",
            "Replace X with one option letter from A through D.",
            "Do not write anything after the FINAL_CHOICE line.",
        ]
    elif query.dataset in {"gsm8k", "math"}:
        answer_contract = [
            "Your final output line must have exactly this form: FINAL_ANSWER: <answer>",
            "Keep the complete final answer on that single line.",
            "Do not write anything after the FINAL_ANSWER line.",
        ]
    else:
        raise ValueError(
            "answer_last_bounded prompt mode supports only gsm8k, "
            "arc_challenge and math; "
            f"received {query.dataset!r}"
        )
    return "\n".join(
        [
            query.prompt.rstrip(),
            "Output format:",
            "First solve the problem in at most six short numbered reasoning steps.",
            *answer_contract,
        ]
    )


def _render_answer_last_stress_prompt(query: Query) -> str:
    """Use a strict line budget and plain-text final marker for stress testing."""
    if query.dataset == "arc_challenge":
        marker = "FINAL_CHOICE"
        final_form = "FINAL_CHOICE: X"
        answer_detail = ["Replace X with one option letter from A through D."]
    elif query.dataset in {"gsm8k", "math"}:
        marker = "FINAL_ANSWER"
        final_form = "FINAL_ANSWER: <answer>"
        answer_detail = []
    else:
        raise ValueError(
            "answer_last_stress prompt mode supports only gsm8k, "
            "arc_challenge and math; "
            f"received {query.dataset!r}"
        )
    return "\n".join(
        [
            query.prompt.rstrip(),
            "Output format:",
            "Solve using no more than six numbered lines.",
            "Each numbered line must contain exactly one short sentence.",
            "Do not continue reasoning after line 6.",
            "Immediately after the last reasoning line, output your best answer even if uncertain.",
            f"The final line must be plain text exactly in this form: {final_form}",
            *answer_detail,
            f"Do not put the {marker} line inside LaTeX, a code block, brackets, or a list item.",
            f"Do not write anything after the {marker} line.",
        ]
    )


def _apply_prompt_mode(
    name: str,
    queries: list[Query],
    prompt_mode: str | None,
) -> list[Query]:
    if prompt_mode is None or name == "supergpqa":
        return queries
    renderers = {
        "answer_first": _render_answer_first_prompt,
        "answer_last_bounded": _render_answer_last_bounded_prompt,
        "answer_last_stress": _render_answer_last_stress_prompt,
    }
    if prompt_mode not in renderers:
        raise ValueError(
            f"unknown prompt mode {prompt_mode!r} for {name}; "
            f"expected one of {list(renderers)}"
        )
    return [
        replace(query, prompt=renderers[prompt_mode](query))
        for query in queries
    ]


def _load_supergpqa_selection(
    limit: int,
    cache_dir: Path,
    selection_manifest: str | Path | None,
    selection_split: str | None,
    prompt_mode: str | None,
) -> list[Query] | None:
    """Load only a frozen UUID split from the pinned local SuperGPQA source."""
    if selection_manifest is None or selection_split is None:
        raise ValueError(
            "supergpqa requires dataset_selection_manifest and dataset_selection_split"
        )
    manifest_path = Path(selection_manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"SuperGPQA selection manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = HF_SPECS["supergpqa"]
    if manifest.get("revision") != spec["revision"]:
        raise ValueError("SuperGPQA selection revision does not match the pinned loader")
    splits = manifest.get("splits") or {}
    if selection_split not in splits:
        raise ValueError(
            f"unknown SuperGPQA selection split {selection_split!r}; "
            f"expected one of {list(splits)}"
        )

    raw_path = (
        cache_dir
        / spec["cache_folder"]
        / spec["revision"]
        / spec["source_file"]
    )
    if not raw_path.exists():
        log.warning("pinned SuperGPQA source is missing: %s", raw_path)
        return None
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if source_hash != manifest.get("source_sha256"):
        raise ValueError("SuperGPQA source hash does not match the frozen selection")

    selected = splits[selection_split]
    ordered_ids = [
        str(qid)
        for difficulty in ("easy", "medium", "hard")
        for qid in selected.get(difficulty, ())
    ]
    selected_id_set = set(ordered_ids)
    if limit < len(ordered_ids):
        raise ValueError(
            f"items_per_dataset={limit} would truncate the frozen SuperGPQA "
            f"selection of {len(ordered_ids)} questions"
        )
    rows_by_id: dict[str, dict] = {}
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("uuid", ""))
            if qid in selected_id_set:
                rows_by_id[qid] = row
    missing = [qid for qid in ordered_ids if qid not in rows_by_id]
    if missing:
        raise ValueError(f"frozen SuperGPQA UUIDs missing from source: {missing[:3]}")

    difficulty_by_id = {
        str(qid): difficulty
        for difficulty, ids in selected.items()
        for qid in ids
    }
    return [
        Query(
            qid=f"supergpqa-{qid}",
            t=0,
            dataset="supergpqa",
            domain=spec["domain"],
            prompt=_render_supergpqa_prompt(
                rows_by_id[qid], prompt_mode or "choice_last"
            ),
            gold_answer=str(rows_by_id[qid]["answer_letter"]).strip().upper(),
            difficulty=difficulty_by_id[qid],
        )
        for qid in ordered_ids
    ]


def _load_from_hub(
    name: str,
    limit: int,
    cache_dir: Path,
    selection_manifest: str | Path | None = None,
    selection_split: str | None = None,
    prompt_mode: str | None = None,
) -> list[Query] | None:
    if name == "supergpqa":
        return _load_supergpqa_selection(
            limit, cache_dir, selection_manifest, selection_split, prompt_mode
        )
    if sys.platform == "win32":
        try:
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass
    try:
        from datasets import load_dataset
    except Exception as exc:
        log.warning("`datasets` import failed for %s (%s)", name, exc)
        return None

    spec = HF_SPECS[name]
    try:
        if name == "math":
            subsets = [
                (
                    config,
                    load_dataset(
                        spec["path"],
                        config,
                        split=spec["split"],
                        cache_dir=str(cache_dir),
                        revision=spec.get("revision"),
                    ),
                )
                for config in spec["name"]
            ]
            indexed_rows = []
            max_size = max(len(subset) for _config, subset in subsets)
            for row_index in range(max_size):
                for config, subset in subsets:
                    if row_index < len(subset):
                        indexed_rows.append((f"{config}-{row_index:05d}", subset[row_index]))
                        if len(indexed_rows) >= limit:
                            break
                if len(indexed_rows) >= limit:
                    break
        else:
            ds = load_dataset(
                spec["path"],
                spec["name"],
                split=spec["split"],
                cache_dir=str(cache_dir),
                revision=spec.get("revision"),
            )
            indexed_rows = [
                (f"{row_index:05d}", row)
                for row_index, row in enumerate(ds)
                if row_index < limit
            ]
    except Exception as exc:  # network, auth, schema change
        log.warning("hub load failed for %s (%s); using synthetic corpus", name, exc)
        return None

    queries: list[Query] = []
    for row_id, row in indexed_rows:
        if name == "arc_challenge":
            try:
                prompt, gold = _arc_prompt_and_gold(row)
            except ValueError as exc:
                log.warning("skipping malformed ARC-Challenge row %s: %s", row_id, exc)
                continue
            raw_answer = str(row[spec["answer"]])
        else:
            prompt = str(row[spec["question"]])
            raw_answer = str(row[spec["answer"]])
            gold = _extract_gold(name, raw_answer)
        if not prompt or not gold:
            continue
        queries.append(
            Query(
                qid=f"{name}-{row_id}",
                t=0,
                dataset=name,
                domain=spec["domain"],
                prompt=prompt,
                gold_answer=gold,
                solution_steps=_solution_step_count(name, raw_answer),
                difficulty_rating=_optional_float(row, "rating"),
                difficulty_rating_std=_optional_float(row, "rating_std"),
                difficulty_quantile=_optional_float(row, "rating_quantile"),
                difficulty_model_avg_acc=_optional_float(row, "model_avg_acc"),
            )
        )
    return queries


def _optional_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _synthetic(name: str, limit: int, seed: int) -> list[Query]:
    """Schema-identical stand-in with a known ground truth.

    Arithmetic items so that correctness is objectively checkable without a
    model: the point of this corpus is to exercise the pipeline, not to make a
    scientific claim.
    """
    rng = random.Random(seed + hash(name) % 10_000)
    domain = HF_SPECS[name]["domain"]
    queries = []
    for i in range(limit):
        a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(1, 12)
        prompt = (
            f"A workshop has {a} boxes with {b} parts each. "
            f"{c} parts are defective and removed. How many usable parts remain?"
        )
        gold = str(a * b - c)
        synthetic_quantile = (
            (i + 0.5) / max(limit, 1) if name.startswith("e2h_") else None
        )
        queries.append(
            Query(
                qid=f"{name}-syn-{i:05d}",
                t=0,
                dataset=name,
                domain=domain,
                prompt=prompt,
                gold_answer=gold,
                solution_steps=2,
                difficulty_rating=synthetic_quantile,
                difficulty_rating_std=0.0 if synthetic_quantile is not None else None,
                difficulty_quantile=synthetic_quantile,
                difficulty_model_avg_acc=(
                    1.0 - synthetic_quantile
                    if synthetic_quantile is not None else None
                ),
            )
        )
    return queries


def _tagged_copy(q: Query, difficulty: str) -> Query:
    return Query(
        qid=q.qid,
        t=0,
        dataset=q.dataset,
        domain=q.domain,
        prompt=q.prompt,
        gold_answer=q.gold_answer,
        difficulty=difficulty,
        paraphrases=q.paraphrases,
        solution_steps=q.solution_steps,
        difficulty_rating=q.difficulty_rating,
        difficulty_rating_std=q.difficulty_rating_std,
        difficulty_quantile=q.difficulty_quantile,
        difficulty_model_avg_acc=q.difficulty_model_avg_acc,
    )


def tag_difficulty_terciles(
    queries: list[Query], seed: int = 42
) -> dict[str, list[Query]]:
    """Split a query pool into Easy / Medium / Hard terciles (E2H design).

    GSM8K has no official difficulty labels, so we use a reproducible weighted
    proxy led by reference-solution operation count, with operand count, prompt
    length and gold magnitude as lower-weight tie breakers.

    This is a *composition control tool*, not a research claim about difficulty
    — the revised design uses difficulty only to create controlled input-
    composition shifts, and reports the proxy openly.

    Returns {"easy": [...], "medium": [...], "hard": [...]}, each shuffled with
    the given seed so within-tercile order carries no difficulty signal.
    """
    import re as _re

    def complexity(q: Query) -> tuple[float, str]:
        n_len = len(q.prompt)
        n_operands = len(_re.findall(r"\d+", q.prompt))
        try:
            gold_mag = abs(float(_re.sub(r"[^\d.\-]", "", q.gold_answer) or 0))
        except ValueError:
            gold_mag = 0.0
        steps = q.solution_steps if q.solution_steps is not None else n_operands
        score = 4.0 * steps + 1.5 * n_operands + 0.01 * n_len + 0.001 * min(gold_mag, 10_000)
        return (score, q.qid)

    ranked = sorted(queries, key=complexity)
    third = len(ranked) // 3
    buckets = {
        "easy":   ranked[:third],
        "medium": ranked[third:2 * third],
        "hard":   ranked[2 * third:],
    }
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)
    tagged = {}
    for diff, items in buckets.items():
        tagged[diff] = [_tagged_copy(q, diff) for q in items]
    return tagged


def tag_difficulty_step_bands(
    queries: list[Query], seed: int = 42
) -> dict[str, list[Query]]:
    """Assign model-independent GSM8K bands from reference solution steps.

    Easy contains 1-2 annotated operations, Medium 3-4, and Hard 5 or
    more. Items without a positive reference-step count are excluded rather
    than assigned through a model-dependent outcome or an opaque fallback.
    """
    buckets = {"easy": [], "medium": [], "hard": []}
    for q in queries:
        steps = q.solution_steps
        if steps is None or steps < 1:
            continue
        if steps <= 2:
            difficulty = "easy"
        elif steps <= 4:
            difficulty = "medium"
        else:
            difficulty = "hard"
        buckets[difficulty].append(q)

    missing = [name for name, items in buckets.items() if not items]
    if missing:
        raise ValueError(
            "solution_steps difficulty scheme produced empty tier(s): "
            + ", ".join(missing)
        )

    rng = random.Random(seed)
    tagged: dict[str, list[Query]] = {}
    for difficulty, items in buckets.items():
        rng.shuffle(items)
        tagged[difficulty] = [_tagged_copy(q, difficulty) for q in items]
    return tagged


def tag_difficulty_supergpqa(
    queries: list[Query], seed: int = 42
) -> dict[str, list[Query]]:
    """Use the frozen official SuperGPQA Easy/Middle/Hard labels."""
    buckets: dict[str, list[Query]] = {"easy": [], "medium": [], "hard": []}
    for query in queries:
        if query.difficulty in buckets:
            buckets[query.difficulty].append(query)
    missing = [name for name, items in buckets.items() if not items]
    if missing:
        raise ValueError(
            "SuperGPQA selection produced empty tier(s): " + ", ".join(missing)
        )
    rng = random.Random(seed)
    tagged: dict[str, list[Query]] = {}
    for difficulty, items in buckets.items():
        rng.shuffle(items)
        tagged[difficulty] = [_tagged_copy(query, difficulty) for query in items]
    return tagged


def e2h_partition(qid: str) -> str:
    """Stable 40/60 item split fixed independently of campaign seeds."""
    value = int(hashlib.sha256(f"e2h-v1:{qid}".encode()).hexdigest()[:8], 16)
    return "calibration" if value % 10 < 4 else "holdout"


def e2h_refined_partition(qid: str) -> str:
    """Refine the untouched 60% holdout into two fixed, disjoint halves."""
    if e2h_partition(qid) == "calibration":
        return "calibration"
    value = int(hashlib.sha256(f"e2h-v2:{qid}".encode()).hexdigest()[:8], 16)
    return "range_calibration" if value % 10 < 5 else "final_holdout"


def _e2h_partition_matches(qid: str, partition: str) -> bool:
    valid = {
        "all", "calibration", "holdout", "range_calibration", "final_holdout"
    }
    if partition not in valid:
        raise ValueError(
            "difficulty_partition must be 'all', 'calibration', 'holdout', "
            "'range_calibration', or 'final_holdout'"
        )
    if partition == "all":
        return True
    if partition in {"calibration", "holdout"}:
        return e2h_partition(qid) == partition
    return e2h_refined_partition(qid) == partition


def _normalized_e2h_quantile(q: Query) -> float | None:
    if q.difficulty_quantile is None:
        return None
    quantile = q.difficulty_quantile
    if quantile > 1.0:
        quantile /= 100.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"invalid E2H rating_quantile for {q.qid}: {quantile}")
    return quantile


def tag_difficulty_e2h_quantiles(
    queries: list[Query], seed: int = 42, partition: str = "all"
) -> dict[str, list[Query]]:
    """Use Easy2Hard-Bench's published item quantiles as fixed difficulty bands."""
    buckets: dict[str, list[Query]] = {"easy": [], "medium": [], "hard": []}
    for q in queries:
        quantile = _normalized_e2h_quantile(q)
        if quantile is None or not _e2h_partition_matches(q.qid, partition):
            continue
        difficulty = "easy" if quantile <= 1 / 3 else (
            "medium" if quantile <= 2 / 3 else "hard"
        )
        buckets[difficulty].append(q)

    missing = [name for name, items in buckets.items() if not items]
    if missing:
        raise ValueError(
            "E2H rating quantiles produced empty tier(s): " + ", ".join(missing)
        )
    rng = random.Random(seed)
    tagged: dict[str, list[Query]] = {}
    for difficulty, items in buckets.items():
        rng.shuffle(items)
        tagged[difficulty] = [_tagged_copy(q, difficulty) for q in items]
    return tagged


def tag_difficulty_e2h_extremes(
    queries: list[Query], seed: int = 42, partition: str = "range_calibration"
) -> dict[str, list[Query]]:
    """Use bottom, central, and top 20% bands to avoid model ceiling effects."""
    buckets: dict[str, list[Query]] = {"easy": [], "medium": [], "hard": []}
    for q in queries:
        quantile = _normalized_e2h_quantile(q)
        if quantile is None or not _e2h_partition_matches(q.qid, partition):
            continue
        if quantile <= 0.20:
            buckets["easy"].append(q)
        elif 0.40 <= quantile <= 0.60:
            buckets["medium"].append(q)
        elif quantile >= 0.80:
            buckets["hard"].append(q)

    missing = [name for name, items in buckets.items() if not items]
    if missing:
        raise ValueError(
            "E2H extreme quantiles produced empty tier(s): " + ", ".join(missing)
        )
    rng = random.Random(seed)
    tagged: dict[str, list[Query]] = {}
    for difficulty, items in buckets.items():
        rng.shuffle(items)
        tagged[difficulty] = [_tagged_copy(q, difficulty) for q in items]
    return tagged


def tag_difficulty(
    queries: list[Query], seed: int = 42, scheme: str = "terciles",
    partition: str = "all",
) -> dict[str, list[Query]]:
    """Dispatch to a declared, reproducible difficulty assignment scheme."""
    if scheme == "terciles":
        return tag_difficulty_terciles(queries, seed=seed)
    if scheme == "solution_steps":
        return tag_difficulty_step_bands(queries, seed=seed)
    if scheme == "supergpqa_official":
        return tag_difficulty_supergpqa(queries, seed=seed)
    if scheme == "e2h_rating_quantiles":
        return tag_difficulty_e2h_quantiles(
            queries, seed=seed, partition=partition
        )
    if scheme == "e2h_extreme_quantiles":
        return tag_difficulty_e2h_extremes(
            queries, seed=seed, partition=partition
        )
    raise ValueError(
        f"unknown difficulty_scheme {scheme!r}; expected 'terciles', "
        "'solution_steps', 'supergpqa_official', 'e2h_rating_quantiles', or "
        "'e2h_extreme_quantiles'"
    )


def difficulty_proxy_summary(
    queries: list[Query], seed: int = 42, scheme: str = "terciles",
    partition: str = "all",
) -> dict:
    """Auditable tier statistics; no model outcomes are inferred here."""
    import re
    tiers = tag_difficulty(
        queries, seed=seed, scheme=scheme, partition=partition
    )
    summary = {
        "scheme": scheme,
        "partition": partition,
        "n_input": len(queries),
        "n_excluded": len(queries) - sum(len(items) for items in tiers.values()),
    }
    for name, items in tiers.items():
        steps = [q.solution_steps for q in items if q.solution_steps is not None]
        operands = [len(re.findall(r"\d+", q.prompt)) for q in items]
        ratings = [q.difficulty_rating for q in items if q.difficulty_rating is not None]
        quantiles = [
            q.difficulty_quantile
            for q in items if q.difficulty_quantile is not None
        ]
        summary[name] = {
            "n": len(items),
            "mean_solution_steps": float(sum(steps) / len(steps)) if steps else None,
            "coverage_solution_steps": len(steps) / max(len(items), 1),
            "mean_operands": float(sum(operands) / len(operands)) if operands else None,
            "mean_prompt_length": float(sum(map(lambda q: len(q.prompt), items)) / len(items))
            if items else None,
            "mean_difficulty_rating": (
                float(sum(ratings) / len(ratings)) if ratings else None
            ),
            "mean_difficulty_quantile": (
                float(sum(quantiles) / len(quantiles)) if quantiles else None
            ),
        }
    step_means = [summary[name]["mean_solution_steps"] for name in ("easy", "medium", "hard")]
    summary["monotonic_solution_steps"] = (
        all(value is not None for value in step_means)
        and step_means[0] < step_means[1] < step_means[2]
    )
    return summary


def load_dataset_queries(
    name: str,
    limit: int = 500,
    cache_dir: str | Path = "data/raw",
    seed: int = 42,
    allow_synthetic: bool = True,
    selection_manifest: str | Path | None = None,
    selection_split: str | None = None,
    prompt_mode: str | None = None,
) -> LoadedDataset:
    if name not in HF_SPECS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(HF_SPECS)}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_from_hub(
        name,
        limit,
        cache_dir,
        selection_manifest=selection_manifest,
        selection_split=selection_split,
        prompt_mode=prompt_mode,
    )
    source = "huggingface_pinned_selection" if name == "supergpqa" else "huggingface"
    if not queries:
        if not allow_synthetic:
            raise RuntimeError(
                f"could not download {name} and allow_synthetic=False"
            )
        queries = _synthetic(name, limit, seed)
        source = "synthetic"

    queries = _apply_prompt_mode(name, queries, prompt_mode)

    return LoadedDataset(
        name=name,
        queries=queries,
        source=source,
        content_hash=_hash_queries(queries),
    )


def load_all(
    names: list[str],
    limit_per_dataset: int = 500,
    cache_dir: str | Path = "data/raw",
    seed: int = 42,
    allow_synthetic: bool = True,
    selection_manifest: str | Path | None = None,
    selection_split: str | None = None,
    prompt_mode: str | None = None,
) -> dict[str, LoadedDataset]:
    return {
        n: load_dataset_queries(
            n,
            limit_per_dataset,
            cache_dir,
            seed,
            allow_synthetic=allow_synthetic,
            selection_manifest=selection_manifest,
            selection_split=selection_split,
            prompt_mode=prompt_mode,
        )
        for n in names
    }


def write_manifest(datasets: dict[str, LoadedDataset], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                n: {
                    "source": d.source,
                    "n_items": len(d.queries),
                    "content_hash": d.content_hash,
                }
                for n, d in datasets.items()
            },
            indent=2,
        )
    )
