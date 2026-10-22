"""Offline diagnostics; a post-hoc sampling rule is not a frozen gate pass."""

import math
import re
from collections import Counter

from reasoning_monitor.dialogue_experiment import messages_sha256
from reasoning_monitor.dialogue_six_metrics import call_key, score_cell
from reasoning_monitor.metrics.dimensions import answers_equal, normalize_answer


def audit_probe(records, jobs, fingerprint, scorers, k=5):
    expected = {call_key(j): j for j in jobs}
    if len(records) != len(expected) or {call_key(r) for r in records} != set(expected):
        raise ValueError('Incomplete, duplicate or unexpected probe records')
    groups = {}
    for r in records:
        job = expected[call_key(r)]
        if (r['config_hash'] != fingerprint
                or messages_sha256(r['messages']) != job['messages_sha256']
                or r['messages_sha256'] != job['messages_sha256']
                or r['gold_answer'] != job['gold_answer']
                or r['max_tokens'] != job['max_tokens']):
            raise ValueError('Probe provenance mismatch')
        groups.setdefault((r['model'], r['dialogue_id'], r['turn']), []).append(r)
    cells = []
    for (model, dialogue, turn), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['call_index'])
        valid = all(r['ok'] and r.get('finish_reason') != 'length'
                    and r['n_tokens'] < r['max_tokens'] for r in rows)
        markers = [re.fullmatch(r'FINAL_ANSWER: (-?\d+)', r['raw_text'].strip().splitlines()[-1])
                   if r['raw_text'].strip() else None for r in rows]
        agreement = all(m is not None and answers_equal(m.group(1), r['answer'], 'dialogue_integer')
                        for m, r in zip(markers, rows, strict=True))
        scores, status = score_cell(rows, scorers, k, 2, [], [])
        # Sparse turns cannot supply a causal five-turn ES window; no paraphrases exist.
        scores.update(RS=float('nan'), ES=float('nan'))
        status.update(RS='no_paraphrase_samples', ES='sparse_turns_no_window')
        if not agreement:
            for metric in ('CQ', 'CS'):
                scores[metric], status[metric] = float('nan'), 'answer_contract_failed'
        counts = Counter(normalize_answer(r['answer'], 'dialogue_integer') for r in rows)
        cells.append(dict(model=model, dialogue_id=dialogue, turn=turn, n=len(rows),
                          raw_unique=len({r['raw_text'] for r in rows}),
                          normalized_answer_counts=dict(counts),
                          unique_answers=len(counts),
                          correct_samples=sum(answers_equal(r['answer'], r['gold_answer'],
                                                           'dialogue_integer') for r in rows),
                          operational_valid=valid and agreement,
                          scores={m: float(v) if math.isfinite(v) else None for m, v in scores.items()},
                          status=status))
    models = {}
    for model in sorted({c['model'] for c in cells}):
        group = [c for c in cells if c['model'] == model]
        models[model] = dict(
            operational_pass=all(c['operational_valid'] for c in group),
            diverse_cells=sum(c['raw_unique'] >= 2 for c in group),
            posthoc_acquisition_readiness=all(c['operational_valid'] for c in group)
            and any(c['raw_unique'] >= 2 for c in group))
    return dict(cells=cells, models=models, no_model_calls=True, exploratory=True,
                rule_version='sampling-readiness-v2-posthoc',
                rule='All cells operationally valid; at least one diverse cell per model. '
                     'Do not require diversity within every question.',
                limitations=['Rule revised after inspecting this probe; not independent validation.',
                             'No proof of provider seed forwarding or stochastic independence.',
                             'No early-warning inference from two sampled turns in one dialogue.',
                             'Original gate and paid execution authorization remain unchanged.'])
