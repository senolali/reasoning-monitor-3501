"""Fixed 336-call calibration pilot; default mode never calls a remote model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dialogue_six_metrics import digest, load_plan, plot_metrics  # noqa: E402

from reasoning_monitor.dialogue_six_metrics import (  # noqa: E402
    METRICS,
    StrictScorers,
    call_key,
    execute_calls,
    score_records,
)
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402

AUDIT_SHA = '253fb30c3796bca4e118aa0a39d1a22318480dd7c8065a0745e36f3e3e63c430'
OUTPUT = ROOT / 'results/dialogue_six_metrics_mini_v1'


def mini_plan():
    config, jobs, parent_hash = load_plan()
    audit_path = ROOT / 'results/dialogue_six_metrics_sampling_audit/audit.json'
    if digest(audit_path) != AUDIT_SHA:
        raise ValueError('Sampling audit changed or is missing')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for path, sha in audit['source_sha256'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('Sampling evidence changed')
    if not all(audit['models'][m['name']]['posthoc_acquisition_readiness'] for m in config['models']):
        raise ValueError('Sampling readiness unavailable')
    jobs = [j for j in jobs if j['dialogue_id'] == 'cdr-v14-budget-002']
    if len(jobs) != 336 or len({call_key(j) for j in jobs}) != 336:
        raise ValueError('Mini plan must have exactly 336 distinct calls')
    protocol = dict(protocol_id='six-metric-mini-v1', parent_hash=parent_hash,
                    audit_sha256=AUDIT_SHA, dialogue='cdr-v14-budget-002', calls=336,
                    k=5, paraphrases=2, temperature=0.7, conditions=config['conditions'],
                    maximum_output_tokens=sum(j['max_tokens'] for j in jobs),
                    exploratory=True, sampling_rule='sampling-readiness-v2-posthoc',
                    stop_rule='No automatic expansion or held-out access; inspect all six metrics.',
                    warning='Previously used calibration dialogue, not an independent holdout.')
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return config, jobs, protocol, fingerprint


def assess(frame):
    events = []
    for (model, condition), rows in frame.groupby(['model', 'condition']):
        rows = rows.sort_values('turn')
        failed = rows[rows.CQ == 0]
        turn = int(failed.iloc[0].turn) if len(failed) else None
        previous = rows[rows.turn == turn - 1] if turn is not None else rows.iloc[:0]
        values = {m: (float(previous.iloc[0][m]) if len(previous)
                      and previous[m].notna().iloc[0] else None) for m in METRICS}
        events.append(dict(model=model, condition=condition, first_error_turn=turn,
                           immediately_preceding_metrics=values,
                           outcome='error_observed' if turn is not None else 'no_error_observed'))
    return dict(no_model_calls=True, exploratory=True, early_warning_established=False,
                note='Descriptive first-error alignment only; no fitted thresholds or significance claims. '
                     'CQ/RS/ES require gold; ES turns 1-4 are warmup. '
                     'No error means warning was not testable, not that the method failed.',
                events=events,
                status_counts={m: frame[f'{m}_status'].value_counts().to_dict() for m in METRICS},
                distinct_values={m: int(frame[m].nunique()) for m in METRICS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--score-only', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config, jobs, protocol, fingerprint = mini_plan()
    print(json.dumps(dict(**protocol, config_hash=fingerprint), indent=2), flush=True)
    if not any(vars(args).values()):
        print('DRY RUN: no remote calls or files written.')
        return
    scorer = StrictScorers(with_ss=True)
    if args.preflight:
        print('PASS: fixed 336-call plan, source hashes, NLI and BERTScore ready. No remote calls.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'records.jsonl'
    if args.score_only:
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        expected = {call_key(j): j['messages_sha256'] for j in jobs}
        if len(records) != 336 or {call_key(r) for r in records} != set(expected):
            raise ValueError('Incomplete or duplicate records; resume the same execution command')
        if any(r['config_hash'] != fingerprint or r['messages_sha256'] != expected[call_key(r)]
               for r in records):
            raise ValueError('Record provenance mismatch')
    else:
        (OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        completed = len(path.read_text(encoding='utf-8').splitlines()) if path.exists() else 0
        progress = ProgressReporter(total=336, label='Mini pilot', completed=completed)
        clients = {m['name']: build_client(m) for m in config['models']}
        records = execute_calls(jobs, clients, path, fingerprint, progress.print)
        progress.print(len(records), final=True)
    frame = score_records(records, scorer, 5, 2)
    frame.to_csv(OUTPUT / 'six_metrics.csv', index=False)
    plot_metrics(frame, OUTPUT)
    report = assess(frame)
    report.update(records_sha256=digest(path), config_hash=fingerprint)
    (OUTPUT / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
