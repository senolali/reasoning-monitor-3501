"""Model clients.

Two backends behind one protocol:

* `OpenAICompatibleClient` — any /v1/chat/completions endpoint (OpenAI, ASU
  CreateAI, DeepSeek, a local vLLM server). This is what the real experiment
  runs on.
* `SimulatedClient` — a deterministic generator whose accuracy depends on the
  query's domain and difficulty. It exists so the pipeline is runnable and
  testable without API keys, budget, or network, and so CI can assert
  end-to-end behaviour. It is NOT a model of any real system: its numbers
  answer "does the monitoring machinery detect a shift that is there", not
  "how good is GPT-4o-mini".

Every result carries the backend name into the manifest so simulated and live
runs can never be silently pooled.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Sequence
from typing import Protocol

from reasoning_monitor.types import Query, Sample


class LLMClient(Protocol):
    name: str
    backend: str

    def generate(self, prompt: str, seed: int = 0) -> Sample: ...


_BLANK_LINE = re.compile(r"\n\s*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=(?:[#*\-]|\d+[.)]|[A-Z]))")
_LIST_ITEM = re.compile(r"^(?:[#*\-]|\d+[.)])\s+")
_FINAL_CHOICE = re.compile(r"(?im)^\s*FINAL_CHOICE\s*:\s*([A-J])\b")
_FINAL_ANSWER = re.compile(r"(?im)^\s*FINAL_ANSWER\s*:\s*(\S.*?)\s*$")
_TYPED_ANSWER_LINE = re.compile(
    r"(?im)^[ \t]*(?:FINAL_CHOICE|FINAL_ANSWER)[ \t]*:[^\r\n]*(?:\r?\n|$)"
)


def reasoning_steps_from_text(text: str) -> list[str]:
    """Split a completion into complete, structure-aware reasoning segments.

    Paragraphs are the primary units. Standalone display-math paragraphs are
    attached to the preceding explanation, and sentence splitting is used only
    when the provider returned a single unstructured paragraph. No segment cap
    is applied, so downstream LS/SS scorers receive the complete trace.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = _TYPED_ANSWER_LINE.sub("", normalized).strip()
    if not normalized:
        return []

    segments: list[str] = []
    for block in _BLANK_LINE.split(normalized):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) > 1 and all(_LIST_ITEM.match(line) for line in lines):
            segments.extend(lines)
            continue
        value = " ".join(lines)
        if not value:
            continue
        is_display_math = (
            value.startswith(r"\[")
            and value.endswith(r"\]")
            or value.startswith("$$")
            and value.endswith("$$")
        )
        if is_display_math and segments:
            segments[-1] = f"{segments[-1]} {value}"
        else:
            segments.append(value)

    if len(segments) == 1:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(segments[0])
            if sentence.strip()
        ]
        if len(sentences) > 1:
            segments = sentences
    return segments


def final_answer_from_text(text: str) -> str:
    """Extract typed anchored answers before falling back to legacy markers."""
    choices = _FINAL_CHOICE.findall(text)
    if choices:
        return choices[-1].upper()
    answers = _FINAL_ANSWER.findall(text)
    if answers:
        return answers[-1].strip()
    return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()


def gateway_probe_cases() -> list[tuple[str, dict]]:
    """Minimal request-option probes for OpenAI-compatible gateways.

    The cases isolate each ASU-specific switch before trying the combined
    payload, so a rejected option can be identified without ambiguity.
    """
    prompt_context = {"time": False, "date": False}
    return [
        ("baseline", {}),
        ("disable_enable_history", {"enable_history": False}),
        ("disable_enable_search", {"enable_search": False}),
        ("disable_prompt_context", {"enhance_prompt": prompt_context}),
        (
            "disable_all",
            {
                "enable_history": False,
                "enable_search": False,
                "enhance_prompt": prompt_context,
            },
        ),
    ]


class SimulatedClient:
    """Deterministic pseudo-model with configurable competence per condition.

    Accuracy is a function of (domain, difficulty), so a scenario that shifts
    the input distribution mechanically shifts observed quality — which is
    exactly the signal the detectors are supposed to find. Determinism comes
    from hashing (model, prompt, seed): the same input always yields the same
    output, across processes and machines.
    """

    backend = "simulated"

    def __init__(
        self,
        name: str,
        competence: dict[str, float] | None = None,
        consistency: float = 0.9,
        base_tokens: int = 180,
        seed: int = 0,
    ):
        self.name = name
        self.competence = competence or {
            # E2H-GSM8K design: graded easy > medium > hard within math domain
            "math:base": 0.82,
            "math:easy": 0.90,
            "math:medium": 0.72,
            "math:hard": 0.48,
            "math:mixed": 0.70,
            "science:base": 0.61,
            "science:easy": 0.70,
            "science:medium": 0.55,
            "science:hard": 0.42,
            "commonsense:base": 0.68,
            "commonsense:easy": 0.75,
            "commonsense:medium": 0.60,
            "commonsense:hard": 0.48,
        }
        self.consistency = consistency
        self.base_tokens = base_tokens
        self.seed = seed
        self._context: tuple[str, str] = ("math", "base")

    def set_context(self, domain: str, difficulty: str) -> None:
        """The scenario tells the client what kind of item is arriving."""
        self._context = (domain, difficulty)

    def _rng(self, prompt: str, seed: int) -> random.Random:
        key = f"{self.name}|{prompt}|{seed}|{self.seed}".encode()
        return random.Random(int(hashlib.sha256(key).hexdigest()[:16], 16))

    def _accuracy(self) -> float:
        domain, difficulty = self._context
        return self.competence.get(
            f"{domain}:{difficulty}", self.competence.get(f"{domain}:base", 0.6)
        )

    def generate(self, prompt: str, seed: int = 0, gold: str = "") -> Sample:
        rng = self._rng(prompt, seed)
        correct = rng.random() < self._accuracy()
        if correct and gold:
            answer = gold
        else:
            # A wrong-but-plausible answer: perturb the gold value if numeric.
            try:
                base = float(gold) if gold else rng.randint(1, 500)
                delta = rng.choice([-3, -2, -1, 1, 2, 3, 7, -7])
                answer = str(int(base) + delta)
            except ValueError:
                answer = f"option-{rng.randint(1, 4)}"

        n_steps = rng.randint(2, 6)
        # Coherence degrades with difficulty: hard items get more contradiction
        # cues, which is what LS is meant to pick up.
        incoherent = rng.random() < (0.10 if self._context[1] != "hard" else 0.35)
        steps = [
            f"Step {i + 1}: compute intermediate value {rng.randint(1, 99)}"
            for i in range(n_steps)
        ]
        if incoherent and n_steps >= 2:
            steps[-1] = "However, this does not follow from the previous result"

        tokens = int(self.base_tokens * rng.uniform(0.6, 1.8))
        if self._context[1] == "hard":
            tokens = int(tokens * 1.6)
        raw_text = ". ".join(steps) + f". Answer: {answer}"
        return Sample(
            answer=answer,
            reasoning_steps=steps,
            latency_s=round(rng.uniform(0.4, 2.5), 3),
            n_tokens=tokens,
            raw_text=raw_text,
        )


class OpenAICompatibleClient:
    """Live backend for any OpenAI-compatible /chat/completions endpoint.

    Note on the K repeats: `generate(..., seed=k)` varies only the *caller's*
    bookkeeping — this client does not put `seed` in the request body, because
    not every gateway accepts it.  With `temperature=0` the K repeats are
    therefore identical by construction and CS/SS collapse to 1.0.  That is a
    protocol decision, not a default to inherit silently, so
    `ReasoningMonitor.preflight()` reports it and refuses it in
    publication_mode.  See DEVIATIONS §24.
    """

    backend = "openai_compatible"

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float | None = 0.0,
        max_tokens: int | None = 512,
        timeout: int = 120,
        extra_headers: dict[str, str] | None = None,
        request_options: dict | None = None,
        send_seed: bool = False,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        rate_limit_backoff: float = 0.0,
        system_prompt: str = "Reason step by step. End with 'Answer: <value>'.",
    ):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.request_options = request_options or {}
        self.send_seed = send_seed
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # Opt-in minimum wait after HTTP 429 (gateway TPM windows are ~60 s);
        # 0 keeps the historical exponential schedule unchanged.
        self.rate_limit_backoff = rate_limit_backoff
        self.system_prompt = system_prompt
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} is not set; export it or use the simulated backend"
            )

    def generate(self, prompt: str, seed: int = 0, gold: str = "") -> Sample:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.generate_messages(messages, seed=seed, gold=gold)

    def generate_messages(
        self,
        messages: Sequence[dict[str, str]],
        seed: int = 0,
        gold: str = "",
    ) -> Sample:
        """Generate from an explicit caller-managed conversation history."""
        import urllib.error
        import urllib.request

        normalized_messages = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant"} or not content:
                raise ValueError("messages require non-empty system/user/assistant entries")
            normalized_messages.append({"role": role, "content": content})
        if not normalized_messages or normalized_messages[-1]["role"] != "user":
            raise ValueError("messages must end with the current user turn")

        body = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
            **self.request_options,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.send_seed:
            body["seed"] = int(seed)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        t0 = time.time()
        payload = None
        last_error = "unknown error"
        transient_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            status = None
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                status = exc.code
                try:
                    detail = exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    detail = ""
                detail = detail[:1000]
                last_error = f"HTTP {exc.code}: {exc.reason}"
                if detail:
                    last_error = f"{last_error}; body={detail}"
                transient = exc.code in transient_statuses
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                transient = True
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = f"invalid response: {type(exc).__name__}: {exc}"
                transient = False
            if not transient or attempt >= self.max_retries:
                return Sample(
                    answer="",
                    reasoning_steps=(),
                    latency_s=round(time.time() - t0, 3),
                    n_tokens=0,
                    raw_text=f"[CALL_FAILED] {last_error}",
                    ok=False,
                )
            delay = self.backoff_base * (2**attempt)
            if status == 429:
                delay = max(delay, self.rate_limit_backoff)
            time.sleep(delay)

        if payload is None:
            return Sample(
                answer="",
                reasoning_steps=(),
                latency_s=round(time.time() - t0, 3),
                n_tokens=0,
                raw_text=f"[CALL_FAILED] {last_error}",
                ok=False,
            )
        latency = time.time() - t0

        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
            usage = payload.get("usage") or {}
        except (KeyError, IndexError, TypeError) as exc:
            return Sample(
                answer="",
                reasoning_steps=(),
                latency_s=round(latency, 3),
                n_tokens=0,
                raw_text=f"[CALL_FAILED] invalid response structure: {exc}",
                ok=False,
            )
        n_tokens = int(usage.get("completion_tokens") or max(len(text) // 4, 1))
        answer = final_answer_from_text(text)
        return Sample(
            answer=answer,
            reasoning_steps=reasoning_steps_from_text(text),
            latency_s=round(latency, 3),
            n_tokens=n_tokens,
            raw_text=text,
            finish_reason=(str(finish_reason) if finish_reason is not None else None),
        )


def build_client(spec: dict, seed: int = 0) -> LLMClient:
    kind = spec.get("backend", "simulated")
    cfg = {k: v for k, v in spec.items() if k != "backend"}
    if kind == "simulated":
        return SimulatedClient(seed=seed, **cfg)
    if kind == "openai_compatible":
        return OpenAICompatibleClient(**cfg)
    raise ValueError(f"unknown backend: {kind!r}")


def make_paraphrases(query: Query, n: int = 3) -> list[str]:
    """Deterministic meaning-preserving rewrites for the robustness probe.

    Template-based rather than model-generated on purpose: a paraphrase
    produced by an LLM would make RS depend on a second model's behaviour and
    would not be reproducible across runs.
    """
    if n < 0:
        raise ValueError("paraphrase count cannot be negative")
    if query.paraphrases:
        if len(query.paraphrases) < n:
            raise ValueError(
                f"query {query.qid!r} has {len(query.paraphrases)} frozen "
                f"paraphrases but L={n} was requested"
            )
        return [str(text).strip() for text in query.paraphrases[:n]]

    p = query.prompt.strip()
    variants = [
        f"Consider the following problem. {p}",
        f"{p} Please work through this carefully.",
        f"Here is a question that needs solving: {p}",
        f"Solve this: {p}",
    ]
    return variants[:n]
