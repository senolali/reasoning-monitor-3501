"""Step-pair incoherence scorers used by LS.

Two implementations with the same interface:

* `NLICoherence` — the formulation in the proposal: DeBERTa-v3-small scores the
  entailment relation between consecutive reasoning steps and the contradiction
  probability is taken as incoherence. Requires torch + transformers.
* `LexicalCoherence` — a dependency-free fallback: contradiction cues plus
  lexical drift between steps. Weaker, but it keeps the pipeline runnable (and
  the CI green) on a machine without a GPU or a model download.

Which one is active is recorded in the run manifest, because the two are not
comparable and must not be pooled across runs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol


class IncoherenceScorer(Protocol):
    name: str

    def score_pairs(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> list[float]:
        """One incoherence value in [0, 1] per (premise, hypothesis) pair."""
        ...


_NEGATION = re.compile(
    r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|don't|wasn't|weren't)\b"
)
_CONTRAST = re.compile(r"\b(but|however|although|contrary|instead|whereas)\b")
_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class LexicalCoherence:
    """Heuristic incoherence: negation flips, contrast markers, topic drift.

    Honest about what it is — a proxy. It cannot detect a semantic
    contradiction expressed without surface cues, which is exactly what the NLI
    scorer is for.
    """

    name = "lexical-v1"

    def score_pairs(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> list[float]:
        out: list[float] = []
        for p, h in zip(premises, hypotheses, strict=False):
            pl, hl = p.lower(), h.lower()
            score = 0.0

            # A negation appearing on exactly one side of the pair is a weak
            # contradiction signal.
            if bool(_NEGATION.search(pl)) != bool(_NEGATION.search(hl)):
                score += 0.35
            if _CONTRAST.search(hl):
                score += 0.15

            # Numeric disagreement: the same quantity restated differently.
            pn = set(_NUMBER.findall(pl))
            hn = set(_NUMBER.findall(hl))
            if pn and hn and not (pn & hn):
                score += 0.20

            # Topic drift: consecutive steps that share almost no vocabulary.
            pt = set(_TOKEN.findall(pl))
            ht = set(_TOKEN.findall(hl))
            if pt and ht:
                overlap = len(pt & ht) / min(len(pt), len(ht))
                if overlap < 0.15:
                    score += 0.30

            out.append(min(score, 1.0))
        return out


class NLICoherence:
    """DeBERTa-v3-small NLI incoherence, as specified in the proposal.

    Deterministic by construction: eval mode, no grad, fixed dtype, and a batch
    size that does not change the result. Loaded lazily so that importing the
    package does not pull in torch.
    """

    name = "deberta-v3-small-nli"

    def __init__(self, model_id: str = "microsoft/deberta-v3-small", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._tok = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval().to(self.device)

    def score_pairs(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> list[float]:
        self._ensure_loaded()
        torch = self._torch
        with torch.no_grad():
            enc = self._tok(
                list(premises),
                list(hypotheses),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
        # Label order varies by checkpoint; contradiction is index 0 for the
        # MNLI-style heads used here. Recorded in the manifest for auditability.
        return probs[:, 0].tolist()


def build_scorer(kind: str = "auto", **kwargs) -> IncoherenceScorer:
    if kind == "lexical":
        return LexicalCoherence()
    if kind == "nli":
        return NLICoherence(**kwargs)
    if kind == "auto":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return NLICoherence(**kwargs)
        except ImportError:
            return LexicalCoherence()
    raise ValueError(f"unknown coherence scorer: {kind!r}")


# ── Trace similarity scorers for SS ──────────────────────────────────────────

class JaccardTraceSim:
    """Token Jaccard similarity — dependency-free fallback for SS.

    rqeval uses this when bert-score is not installed.
    """
    name = "jaccard"

    def score_pairs(
        self, traces_a: list[str], traces_b: list[str]
    ) -> list[float]:
        scores = []
        for a, b in zip(traces_a, traces_b, strict=False):
            ta = set(a.lower().split())
            tb = set(b.lower().split())
            union = ta | tb
            scores.append(len(ta & tb) / len(union) if union else 1.0)
        return scores


class BertScoreTraceSim:
    """Pairwise BERTScore F1 across reasoning traces (matches rqeval definition).

    Uses distilbert-base-uncased by default (same as rqeval).
    Loaded lazily so importing the package does not pull in torch.
    """
    name = "bertscore"

    def __init__(
        self,
        model_id: str = "distilbert-base-uncased",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._scorer = None

    def _ensure_loaded(self) -> None:
        if self._scorer is not None:
            return
        from bert_score import BERTScorer  # type: ignore
        self._scorer = BERTScorer(
            model_type=self.model_id,
            device=self.device,
            rescale_with_baseline=False,
        )

    def score_pairs(
        self, traces_a: list[str], traces_b: list[str]
    ) -> list[float]:
        self._ensure_loaded()
        _, _, F = self._scorer.score(traces_a, traces_b)
        return F.tolist()


def build_trace_scorer(kind: str = "auto", **kwargs):
    """Factory for SS trace similarity scorers.

    kind='bertscore' : BERTScore (requires pip install bert-score)
    kind='jaccard'   : Token Jaccard (no extra dependencies)
    kind='auto'      : BERTScore if available, else Jaccard
    """
    if kind == "jaccard":
        return JaccardTraceSim()
    if kind == "bertscore":
        return BertScoreTraceSim(**kwargs)
    if kind == "auto":
        try:
            import bert_score  # noqa: F401
            return BertScoreTraceSim(**kwargs)
        except ImportError:
            return JaccardTraceSim()
    raise ValueError(f"unknown trace scorer: {kind!r}")
