"""
reasoning-monitor rqeval integration verification script.

Run this in a fresh environment after:
    pip install rqeval
    pip install -e ".[dev]"

Expected output: all checks PASS.
"""
import sys
import importlib.metadata


def check(label, fn):
    try:
        result = fn()
        print(f"  ✅ {label}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False


print("=" * 60)
print("reasoning-monitor — rqeval integration verification")
print("=" * 60)

all_ok = True

print("\n[1] Package versions")
all_ok &= check("rqeval pip version",
    lambda: importlib.metadata.version("rqeval"))
try:
    import rqeval
    check("rqeval.__version__", lambda: rqeval.__version__)
    check("rqeval.__file__", lambda: rqeval.__file__)
except ImportError as e:
    print(f"  ❌ rqeval import: {e}")
    all_ok = False

print("\n[2] metrics.* importability (installed by rqeval)")
for name, mod in [
    ("accuracy  (CQ)", "metrics.accuracy"),
    ("consistency (CS)", "metrics.consistency"),
    ("robustness  (RS)", "metrics.robustness"),
    ("logical_consistency (LS)", "metrics.logical_consistency"),
    ("efficiency  (ES)", "metrics.efficiency"),
    ("explainability (SS)", "metrics.explainability"),
    ("answer_extraction", "metrics.answer_extraction"),
]:
    all_ok &= check(name, lambda m=mod: __import__(m))

print("\n[3] rqeval_adapters import")
try:
    from reasoning_monitor.metrics.rqeval_adapters import (
        CQAdapter, CSAdapter, RSAdapter, LSAdapter, ESAdapter, SSAdapter
    )
    print("  ✅ All 6 adapters importable")
except ImportError as e:
    print(f"  ❌ {e}")
    all_ok = False

print("\n[4] Single-query sanity check (no model calls)")
try:
    from reasoning_monitor.types import Query, QueryObservation, Sample
    from reasoning_monitor.metrics.rqeval_adapters import CQAdapter

    q = Query(qid="v", t=1, dataset="gsm8k", domain="math",
              prompt="What is 2+2?", gold_answer="4")
    s = Sample(answer="4", reasoning_steps=["2+2=4"], latency_s=0.1, n_tokens=5)
    obs = QueryObservation(query=q, runs=[s, s, s, s, s], paraphrase_runs=[s, s, s])

    m = CQAdapter(window_size=3)
    val = m.per_query_value(obs)
    all_ok &= check("CQAdapter.per_query_value('4'=='4')", lambda: val)
    assert val == 1.0, f"Expected 1.0, got {val}"
    print("  ✅ CQ=1.0 for correct answer")
except Exception as e:
    import traceback
    traceback.print_exc()
    all_ok = False

print("\n[5] RS uses public is_correct (not private _is_correct)")
try:
    from metrics.answer_extraction import is_correct
    assert is_correct("42", "42"), "is_correct should return True"
    assert not is_correct("41", "42"), "is_correct should return False"
    print("  ✅ metrics.answer_extraction.is_correct is public and works")
except Exception as e:
    print(f"  ❌ {e}")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✅ All checks PASSED — rqeval integration ready.")
else:
    print("❌ Some checks FAILED — fix before running experiments.")
    sys.exit(1)
