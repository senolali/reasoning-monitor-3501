"""LHE-RQ-v1: per-conversation early warning on Long-Horizon Execution.

Protocol: ``LHE_RQ_EARLY_WARNING_PROTOCOL.md`` (draft). Data: Sinha et al.
(ICLR 2026), ``arvindh75/Long-Horizon-Execution`` (Apache-2.0): a 100-key
dictionary and 100 independent key sequences with the exact running sum after
every key.

One *conversation* is one dataset sample. At turn t the user gives ``k`` keys;
the model updates the running sum and ends with ``<answer>N</answer>``. Every
turn has exact gold. Conversations never share context.

This module is call-free: data, prompts, parsing, per-turn signals, the
monitor and the evaluation. Acquisition lives in
``scripts/run_lhe_early_warning.py``. Gold is used only for targets and for
evaluating alarms; the monitor never sees it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "LHE-RQ-v1"
DATA_HASHES = {
    "test.jsonl": "111ef6e3850368efd84175576c279aeba36ddfb20da3582c01a078bfa77620c6",
    "dictionary.json": "33d783c31d664d791a6ff08671b86775cac681768d2853987d528c30457c017e",
}
CALIBRATION_IDS = tuple(range(0, 30))
TEST_IDS = tuple(range(30, 100))
HEALTHY_TURNS = 5          # reference period; also the healthy-start check
ONSET_WINDOW = 5           # tau: first t > 5 with <= 3 of turns t..t+4 step-correct
ONSET_MAX_CORRECT = 3
SIGNALS = ("CS", "SS", "PA", "LS")
PRIMARY_CANDIDATES = ("CS", "SS", "PA")

# ── data ────────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(data_dir: Path, verify: bool = True) -> tuple[dict[str, int], list[dict]]:
    """Dictionary and samples; refuses data whose hash differs from the frozen one."""
    data_dir = Path(data_dir)
    if verify:
        for name, expected in DATA_HASHES.items():
            if sha256_file(data_dir / name) != expected:
                raise ValueError(f"{name}: SHA-256 differs from the frozen dataset")
    dictionary = json.loads((data_dir / "dictionary.json").read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in
               (data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return dictionary, samples


def build_conversation(sample: Mapping[str, Any], k: int, horizon: int) -> dict[str, Any]:
    """Group the key sequence into turns of ``k`` keys; gold = running sum after the turn."""
    if k < 1 or horizon < 1 or k * horizon > len(sample["input"]):
        raise ValueError("invalid turn complexity or horizon")
    turns = []
    for t in range(1, horizon + 1):
        lo, hi = (t - 1) * k, t * k
        keys = list(sample["input"][lo:hi])
        delta = int(sum(sample["values"][lo:hi]))
        turns.append({"turn": t, "keys": keys, "delta": delta,
                      "true_sum": int(sample["output"][hi - 1])})
    running = 0
    for turn in turns:                       # self-check against the dataset
        running += turn["delta"]
        if running != turn["true_sum"]:
            raise ValueError(f"sample {sample['id']}: running sum mismatch at turn {turn['turn']}")
    return {"conversation_id": int(sample["id"]), "k": k, "horizon": horizon, "turns": turns}


# ── prompts ─────────────────────────────────────────────────────────────────
# Adapted from the authors' chain-of-thought template (src/experiments/dict_sum/
# exp.py). Recorded deviation: two of their demonstrations carry wrong answers
# ("<answer>f10</answer>"; a group example answering 10 where 9 is correct);
# ours are correct.

_EXAMPLE_DICT = "{'apple': 5, 'banana': 0, 'cherry': 7, 'grape': -4, 'kiwi': 2, 'mango': -1}"
_EXAMPLES_K1 = (
    "**User**: apple\n**Assistant**: The value for 'apple' is 5. Starting from 0, the "
    "running sum is 0 + 5 = 5. <answer>5</answer>\n\n"
    "**User**: banana\n**Assistant**: The value for 'banana' is 0. The current running sum "
    "is 5, so 5 + 0 = 5. <answer>5</answer>\n\n"
    "**User**: cherry\n**Assistant**: The value for 'cherry' is 7. The current running sum "
    "is 5, so 5 + 7 = 12. <answer>12</answer>\n\n"
    "**User**: grape\n**Assistant**: The value for 'grape' is -4. The current running sum "
    "is 12, so 12 + (-4) = 8. <answer>8</answer>\n\n")
_EXAMPLES_GROUP = (
    "**User**: apple, banana\n**Assistant**: Let me look up the values: 'apple' = 5, "
    "'banana' = 0. Starting from 0, I calculate: 0 + 5 + 0 = 5. <answer>5</answer>\n\n"
    "**User**: cherry, grape\n**Assistant**: Let me look up the values: 'cherry' = 7, "
    "'grape' = -4. Current running sum is 5, so: 5 + 7 + (-4) = 8. <answer>8</answer>\n\n"
    "**User**: kiwi, mango\n**Assistant**: Let me look up the values: 'kiwi' = 2, "
    "'mango' = -1. Current running sum is 8, so: 8 + 2 + (-1) = 9. <answer>9</answer>\n\n")

CONTRACT = ("Think step by step briefly, then give the current running total "
            "inside <answer></answer> tags.")

# Direct-answer mode (pilot-2 deviation; the authors' non-chain-of-thought
# template, which is their main turn-scaling setting). No reasoning is written,
# so trace-based SS and LS are not defined in this mode.
DIRECT_CONTRACT = ("Respond only with the current running total inside <answer></answer> "
                   "tags. Do not output any other text.")
_EXAMPLES_DIRECT = (
    "**User**: apple, banana\n**Assistant**: <answer>5</answer>\n\n"
    "**User**: cherry, grape\n**Assistant**: <answer>8</answer>\n\n"
    "**User**: kiwi, mango\n**Assistant**: <answer>9</answer>\n\n")
MODES = ("cot", "direct")


def contract(mode: str = "cot") -> str:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    return CONTRACT if mode == "cot" else DIRECT_CONTRACT


def system_prompt(dictionary: Mapping[str, int], k: int, mode: str = "cot") -> str:
    unit = "keys one at a time" if k == 1 else f"keys in groups of {k}"
    if mode == "direct":
        return ("You are an AI assistant. I will provide you with a dictionary and then give "
                f"you {unit}. Your task is to keep a running total (starting from 0) by adding "
                "the values associated with the keys I provide.\n\n"
                "For each message, respond with the current total enclosed between "
                "<answer></answer> tags.\n\n"
                f"Examples:\nDictionary to maintain: {_EXAMPLE_DICT}\n\n{_EXAMPLES_DIRECT}"
                "Now, here is the actual task. The dictionary to maintain:\n"
                f"{json.dumps(dict(dictionary), ensure_ascii=True)}\n\n"
                "IMPORTANT: DO NOT OUTPUT ANY OTHER TEXT OUTSIDE ANSWER TAGS. Only provide the "
                "final running sum in <answer> tags.")
    contract(mode)
    examples = _EXAMPLES_K1 if k == 1 else _EXAMPLES_GROUP
    return ("You are an AI assistant. I will provide you with a dictionary and then give you "
            f"{unit}. Your task is to keep a running total (starting from 0) by adding the "
            "value associated with each key I provide.\n\n"
            "For each message, think step by step about the calculation, then respond with "
            "the current total enclosed between <answer></answer> tags.\n\n"
            f"Examples:\nDictionary to maintain: {_EXAMPLE_DICT}\n\n{examples}"
            "Now, here is the actual task. The dictionary to maintain:\n"
            f"{json.dumps(dict(dictionary), ensure_ascii=True)}\n\n"
            "IMPORTANT: Always think step by step about each calculation, showing your work. "
            "Only put the final running sum in <answer> tags.")


# Frozen user-turn wordings: index 0 is the primary; 1..L are rewordings.
TURN_WORDINGS = (
    "{keys}",
    "Next {noun}: {keys}",
    "Please add the value{s} of {keys} to the running total.",
)


def turn_text(keys: Sequence[str], variant: int = 0) -> str:
    joined = ", ".join(keys)
    many = len(keys) > 1
    return TURN_WORDINGS[variant].format(keys=joined, noun="keys" if many else "key",
                                         s="s" if many else "")


def pack_messages(system: str, history: Sequence[Mapping[str, str]], text: str,
                  mode: str = "cot") -> list[dict[str, str]]:
    """Packed transcript in one user message (the gateway drops native history)."""
    if history:
        context = ("Conversation so far (oldest first), as JSON:\n"
                   + json.dumps(list(history), ensure_ascii=True))
    else:
        context = "Conversation so far: none."
    return [{"role": "system", "content": system},
            {"role": "user",
             "content": f"{context}\n\nNew message:\n{text}\n\n{contract(mode)}"}]


def prompt_sha256(dictionary: Mapping[str, int], k: int, mode: str = "cot") -> str:
    """Hash of the frozen prompt; identical to the pilot-1 hash for mode="cot"."""
    probe = pack_messages(system_prompt(dictionary, k, mode),
                          [{"role": "user", "content": "u"},
                           {"role": "assistant", "content": "a"}], turn_text(["x"]), mode)
    payload = {"messages": probe, "wordings": TURN_WORDINGS, "contract": contract(mode)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ── parsing ─────────────────────────────────────────────────────────────────

_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_INT = re.compile(r"-?\d[\d,]*")


def parse_answer(text: str) -> tuple[int | None, bool]:
    """(value, strict). Value: the last integer inside the last answer tag.

    strict: exactly one answer tag whose content is a bare integer. Replies
    without a parseable answer give ``None`` and are never imputed.
    """
    tags = _TAG.findall(text or "")
    if not tags:
        return None, False
    content = tags[-1]
    numbers = _INT.findall(content)
    if not numbers:
        return None, False
    value = int(numbers[-1].replace(",", ""))
    strict = len(tags) == 1 and re.fullmatch(r"-?\d+", content.strip()) is not None
    return value, strict


def reasoning_text(text: str) -> str:
    """Trace without the answer tag (for SS and LS)."""
    return _TAG.sub(" ", text or "").strip()


# ── per-turn signals ────────────────────────────────────────────────────────


def _pairwise_agreement(values: Sequence[int | None]) -> float:
    """RQEval CS convention: a missing answer agrees with nothing."""
    pairs = list(combinations(range(len(values)), 2))
    if not pairs:
        return math.nan
    agree = sum(values[i] is not None and values[i] == values[j] for i, j in pairs)
    return agree / len(pairs)


def _entropy(values: Sequence[int | None]) -> float:
    labels = [("v", v) if v is not None else ("missing", i) for i, v in enumerate(values)]
    counts = np.array([labels.count(x) for x in set(labels)], dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def valid_call(row: Mapping[str, Any]) -> bool:
    return bool(row.get("ok")) and not row.get("truncated")


def score_conversation(rows: Sequence[Mapping[str, Any]], conversation: Mapping[str, Any],
                       k_samples: int, l_paraphrases: int, scorers=None) -> list[dict]:
    """Per-turn targets and signals for one (model, conversation).

    ``rows`` are raw call records with ``turn``, ``role`` ("primary", "repeat",
    "paraphrase"), ``call_index`` and ``raw_text``. ``scorers`` is a
    ``StrictScorers`` instance (real NLI/BERTScore) or None, in which case SS
    and LS stay missing. Nothing is imputed.
    """
    by_turn: dict[int, list] = {}
    for row in rows:
        by_turn.setdefault(int(row["turn"]), []).append(row)
    out = []
    previous: int | None = 0
    for spec in conversation["turns"]:
        t = spec["turn"]
        cell = by_turn.get(t, [])
        primary = next((r for r in cell if r["role"] == "primary"), None)
        record = {"turn": t, "true_sum": spec["true_sum"], "delta": spec["delta"]}
        if primary is None or not valid_call(primary):
            record.update(answer=None, strict=False, step_correct=math.nan,
                          prefix_correct=math.nan, **{s: math.nan for s in SIGNALS},
                          entropy=math.nan, n_tokens=math.nan)
            out.append(record)
            previous = None
            continue
        answer, strict = parse_answer(primary["raw_text"])
        step = (math.nan if answer is None or previous is None
                else float(answer == previous + spec["delta"]))
        record.update(answer=answer, strict=strict, step_correct=step,
                      prefix_correct=math.nan if answer is None
                      else float(answer == spec["true_sum"]),
                      n_tokens=float(primary.get("n_tokens", math.nan)))
        runs = sorted([r for r in cell if r["role"] in ("primary", "repeat")],
                      key=lambda r: r["call_index"])
        paras = [r for r in cell if r["role"] == "paraphrase"]
        if len(runs) == k_samples and k_samples >= 2 and all(valid_call(r) for r in runs):
            answers = [parse_answer(r["raw_text"])[0] for r in runs]
            record["CS"] = _pairwise_agreement(answers)
            record["entropy"] = _entropy(answers)
            traces = [reasoning_text(r["raw_text"]) for r in runs]
            if scorers is not None and getattr(scorers, "bert", None) is not None and all(traces):
                pairs = list(combinations(traces, 2))
                record["SS"] = float(np.mean(scorers.similarity([a for a, _ in pairs],
                                                                 [b for _, b in pairs])))
            else:
                record["SS"] = math.nan
        else:
            record.update(CS=math.nan, SS=math.nan, entropy=math.nan)
        if len(paras) == l_paraphrases and l_paraphrases >= 1 and all(valid_call(r) for r in paras):
            para_answers = [parse_answer(r["raw_text"])[0] for r in paras]
            record["PA"] = (math.nan if answer is None else
                            sum(a == answer for a in para_answers) / len(para_answers))
        else:
            record["PA"] = math.nan
        if scorers is not None:
            from reasoning_monitor.models.clients import reasoning_steps_from_text
            record["LS"] = scorers.coherence(reasoning_steps_from_text(reasoning_text(
                primary["raw_text"])))
        else:
            record["LS"] = math.nan
        out.append(record)
        previous = answer
    return out


# ── events ──────────────────────────────────────────────────────────────────


def healthy_start(step: Sequence[float]) -> bool:
    head = [x for x in step[:HEALTHY_TURNS] if not math.isnan(x)]
    return len(head) == HEALTHY_TURNS and sum(head) >= HEALTHY_TURNS - 1


def decline_onset(step: Sequence[float]) -> int | None:
    """tau (1-based turn) or None for a non-declining conversation.

    tau is the first error inside the first window t..t+4 (t > 5) with at most
    3 step-correct turns. Missing turns count as not correct.
    """
    values = [0.0 if math.isnan(x) else x for x in step]
    for start in range(HEALTHY_TURNS, len(values) - ONSET_WINDOW + 1):
        window = values[start:start + ONSET_WINDOW]
        if sum(window) <= ONSET_MAX_CORRECT:
            first_error = next(i for i, v in enumerate(window) if v < 1)
            return start + first_error + 1
    return None


# ── monitor ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MonitorRule:
    signal: str
    direction: int        # +1 alarm on increase, -1 on decrease
    scale: float          # SD of healthy calibration turns
    threshold: float      # CUSUM h
    slack: float = 0.5    # CUSUM k, in SD units


def cusum_alarm(values: Sequence[float], rule: MonitorRule) -> int | None:
    """First alarm turn (1-based) of a one-sided CUSUM; None if no alarm.

    Reference: the conversation's own mean over turns 1..5 (no further warm-up).
    Missing values neither update the statistic nor advance it.
    """
    head = [v for v in values[:HEALTHY_TURNS] if not math.isnan(v)]
    if len(head) < 3 or rule.scale <= 0:
        return None
    mu = float(np.mean(head))
    s = 0.0
    for t in range(HEALTHY_TURNS, len(values)):
        v = values[t]
        if math.isnan(v):
            continue
        z = rule.direction * (v - mu) / rule.scale
        s = max(0.0, s + z - rule.slack)
        if s > rule.threshold:
            return t + 1
    return None


def classify(alarm: int | None, tau: int | None, lead_window: int = 10,
             lag_window: int = 5) -> str:
    if tau is None:
        return "false_alarm" if alarm is not None else "quiet"
    if alarm is None:
        return "miss"
    if alarm < tau - lead_window:
        return "false_alarm"
    if alarm <= tau + lag_window:
        return "detection"
    return "late"


def fit_rule(signal: str, calibration: Sequence[Mapping[str, Any]], far_target: float = 0.10,
             grid: Sequence[float] = tuple(np.round(np.arange(0.5, 15.01, 0.25), 2))
             ) -> MonitorRule:
    """Direction, scale and threshold from calibration conversations only.

    Direction: sign of (mean post-onset - mean pre-onset) pooled over declining
    calibration conversations. Scale: SD of pre-onset turns. Threshold: the
    smallest h whose calibration false-alarm rate is <= far_target.
    """
    pre, post = [], []
    for conv in calibration:
        tau = conv["tau"]
        values = conv["signals"][signal]
        cut = tau - 1 if tau is not None else len(values)
        pre += [v for v in values[:cut] if not math.isnan(v)]
        if tau is not None:
            post += [v for v in values[tau - 1:] if not math.isnan(v)]
    if len(pre) < 10 or len(post) < 5:
        raise ValueError(f"{signal}: too little calibration support")
    direction = 1 if np.mean(post) > np.mean(pre) else -1
    scale = float(np.std(pre, ddof=1)) or 1e-6
    for h in grid:
        rule = MonitorRule(signal, direction, scale, float(h))
        labels = [classify(cusum_alarm(c["signals"][signal], rule), c["tau"]) for c in calibration]
        if np.mean([x == "false_alarm" for x in labels]) <= far_target:
            return rule
    return MonitorRule(signal, direction, scale, float(grid[-1]))


def evaluate(rule: MonitorRule, conversations: Sequence[Mapping[str, Any]],
             oracle: MonitorRule | None = None) -> dict[str, Any]:
    labels, leads = [], []
    for conv in conversations:
        alarm = cusum_alarm(conv["signals"][rule.signal], rule)
        labels.append(classify(alarm, conv["tau"]))
        if oracle is not None and conv["tau"] is not None and alarm is not None:
            o = cusum_alarm(conv["signals"][oracle.signal], oracle)
            if o is not None:
                leads.append(o - alarm)
    declining = [c for c in conversations if c["tau"] is not None]
    return {"signal": rule.signal, "n": len(conversations), "declining": len(declining),
            "detection_rate": (sum(x == "detection" for x in labels) / len(declining)
                               if declining else math.nan),
            "false_alarm_rate": float(np.mean([x == "false_alarm" for x in labels])),
            "labels": labels, "leads_over_oracle": leads,
            "median_lead": float(np.median(leads)) if leads else math.nan}


def permutation_p(rule: MonitorRule, conversations: Sequence[Mapping[str, Any]],
                  n_perm: int = 10_000, seed: int = 20261105) -> float:
    """One-sided p: detection rate with turns permuted within each conversation."""
    observed = evaluate(rule, conversations)["detection_rate"]
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        shuffled = []
        for conv in conversations:
            values = list(conv["signals"][rule.signal])
            rng.shuffle(values)
            shuffled.append({**conv, "signals": {**conv["signals"], rule.signal: values}})
        hits += evaluate(rule, shuffled)["detection_rate"] >= observed
    return (hits + 1) / (n_perm + 1)


# ── budget ──────────────────────────────────────────────────────────────────


def estimate_conversation_tokens(dictionary: Mapping[str, int], conversation: Mapping[str, Any],
                                 calls_per_turn: int, reply_chars: int,
                                 chars_per_token: float = 4.0, mode: str = "cot"
                                 ) -> dict[str, float]:
    """Char/4 proxy for one conversation: input grows with the packed history."""
    system = system_prompt(dictionary, conversation["k"], mode)
    history: list[dict[str, str]] = []
    input_chars = 0
    for spec in conversation["turns"]:
        messages = pack_messages(system, history, turn_text(spec["keys"]), mode)
        input_chars += calls_per_turn * sum(len(m["content"]) for m in messages)
        history += [{"role": "user", "content": turn_text(spec["keys"])},
                    {"role": "assistant", "content": "x" * reply_chars}]
    turns = len(conversation["turns"])
    return {"calls": calls_per_turn * turns,
            "input_tokens": input_chars / chars_per_token,
            "output_tokens": calls_per_turn * turns * reply_chars / chars_per_token}


# ── simulated client (tests and fixture replay only) ───────────────────────


class SimulatedLHEClient:
    """Deterministic stand-in: reads the packed history, adds the values, and
    errs with a probability that grows after a hidden per-conversation onset.
    Before the onset its samples agree; afterwards they diverge, so a working
    pipeline must be able to detect the planted signal.
    """

    def __init__(self, dictionary: Mapping[str, int], onset_turn: int = 20,
                 error_rate_after: float = 0.6):
        self.dictionary = dict(dictionary)
        self.onset_turn = onset_turn
        self.error_rate_after = error_rate_after

    def generate_messages(self, messages, seed: int = 0, gold: str = ""):
        from reasoning_monitor.types import Sample
        content = messages[-1]["content"]
        history_part, _, new = content.partition("\n\nNew message:\n")
        new_text = new.split("\n\n")[0]
        history = []
        if history_part.startswith("Conversation so far (oldest first)"):
            history = json.loads(history_part.split("as JSON:\n", 1)[1])
        previous = 0
        for message in history:
            if message["role"] == "assistant":
                value, _ = parse_answer(message["content"])
                previous = value if value is not None else previous
        keys = [w.strip(" .") for w in re.findall(r"[a-z]{5}", new_text)
                if w.strip(" .") in self.dictionary]
        turn = sum(m["role"] == "user" for m in history) + 1
        rng = random.Random(seed)
        delta = sum(self.dictionary[key] for key in keys)
        answer = previous + delta
        if turn >= self.onset_turn and rng.random() < self.error_rate_after:
            answer += rng.choice([-10, -1, 1, 10])
        text = (f"The values are {', '.join(f'{k} = {self.dictionary[k]}' for k in keys)}. "
                f"The current running sum is {previous}, so {previous} + ({delta}) = {answer}. "
                f"<answer>{answer}</answer>")
        return Sample(answer=str(answer), reasoning_steps=[text], latency_s=0.0,
                      n_tokens=len(text) // 4, raw_text=text, ok=True, finish_reason="stop")
