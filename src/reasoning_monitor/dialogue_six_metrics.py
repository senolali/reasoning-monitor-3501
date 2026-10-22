"""Strict six-metric scoring and independent same-context sample acquisition."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from reasoning_monitor.dialogue_experiment import messages_sha256
from reasoning_monitor.metrics.dimensions import answers_equal
from reasoning_monitor.metrics.rqeval_adapters import _ESMetric
from reasoning_monitor.models.clients import reasoning_steps_from_text
from reasoning_monitor.primary_dialogue_validation import build_primary_validation_messages

METRICS = ('CQ', 'CS', 'RS', 'LS', 'ES', 'SS')


class StrictScorers:
    def __init__(self, with_ss=False):
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import pipeline

        torch.set_num_threads(2)
        self.nli = pipeline('text-classification', model='cross-encoder/nli-deberta-v3-small',
                            device=-1, top_k=None)
        labels = {str(x).lower() for x in self.nli.model.config.id2label.values()}
        if not {'contradiction', 'entailment', 'neutral'} <= labels:
            raise ValueError(f'Unverified NLI labels: {labels}')
        self.bert = None
        if with_ss:
            from bert_score import BERTScorer
            self.bert = BERTScorer(model_type='distilbert-base-uncased', device='cpu',
                                  rescale_with_baseline=False)
            self.similarity(['A person is walking.'], ['A person walks.'])
        self.coherence(['A person is walking.', 'Nobody is walking.'])

    def coherence(self, steps):
        if len(steps) < 2:
            return np.nan
        # Preserve rqeval's consecutive-step [SEP] input and hard-label formula.
        inputs = [f'{a} [SEP] {b}' for a, b in zip(steps, steps[1:], strict=False)]
        results = self.nli(inputs, truncation=True, max_length=512, batch_size=8)
        if len(results) != len(inputs):
            raise ValueError('Incomplete NLI result')
        predictions = [max(row, key=lambda x: x['score'])['label'].lower() for row in results]
        return 1 - sum(x == 'contradiction' for x in predictions) / len(predictions)

    def similarity(self, left, right):
        if self.bert is None:
            raise ValueError('BERTScore not loaded')
        scores = self.bert.score(left, right, batch_size=8)[2]
        return [max(0.0, min(1.0, float(x))) for x in scores]


def score_cell(records, scorers, k, n_paraphrases, token_history, cq_history, window=5):
    runs = sorted([r for r in records if r['role'] in ('primary', 'repeat')],
                  key=lambda r: r['call_index'])
    paras = [r for r in records if r['role'] == 'paraphrase']
    primary = runs[0]
    result = {name: np.nan for name in METRICS}
    status = {name: 'missing_samples' for name in METRICS}

    def valid(r):
        return bool(r['ok']) and r.get('finish_reason') != 'length' and (
            int(r['n_tokens']) < int(r.get('max_tokens', 2048))
        )

    if not valid(primary):
        return result, {name: 'invalid_primary' for name in METRICS}
    cq = float(answers_equal(primary['answer'], primary['gold_answer'], 'dialogue_integer'))
    result['CQ'], status['CQ'] = cq, 'measured'
    steps = reasoning_steps_from_text(primary['raw_text'])
    result['LS'] = scorers.coherence(steps)
    status['LS'] = 'measured' if np.isfinite(result['LS']) else 'insufficient_steps'
    token_history.append(int(primary['n_tokens']))
    cq_history.append(cq)
    del token_history[:-window]
    del cq_history[:-window]
    result['ES'] = float(_ESMetric().compute_per_instance(cq_history, token_history)[-1])
    status['ES'] = 'warmup' if len(token_history) < window else 'measured'
    if len(runs) == k and k >= 2 and all(valid(r) for r in runs):
        pairs = list(combinations(runs, 2))
        result['CS'] = sum(answers_equal(a['answer'], b['answer'], 'dialogue_integer')
                           for a, b in pairs) / len(pairs)
        status['CS'] = 'measured'
        traces = [' '.join(reasoning_steps_from_text(r['raw_text'])) for r in runs]
        if all(traces):
            trace_pairs = list(combinations(traces, 2))
            result['SS'] = float(np.mean(scorers.similarity(
                [a for a, _ in trace_pairs], [b for _, b in trace_pairs])))
            status['SS'] = 'measured'
    if cq == 0:
        status['RS'] = 'C_subset_excluded'
    elif len(paras) == n_paraphrases and n_paraphrases >= 1 and all(valid(r) for r in paras):
        result['RS'] = sum(answers_equal(r['answer'], primary['gold_answer'], 'dialogue_integer')
                           for r in paras) / len(paras)
        status['RS'] = 'measured'
    return result, status


def score_records(records, scorers, k=5, n_paraphrases=2):
    groups = {}
    for record in records:
        key = (record['model'], record['dialogue_id'], record['condition'], record['turn'])
        groups.setdefault(key, []).append(record)
    histories, rows = {}, []
    for key, cell in sorted(groups.items()):
        primary = next((r for r in cell if r['role'] == 'primary'), None)
        if primary is None:
            continue
        history = histories.setdefault(key[:3], ([], []))
        values, status = score_cell(cell, scorers, k, n_paraphrases, *history)
        rows.append(dict(model=key[0], dialogue_id=key[1], condition=key[2], turn=key[3],
                         **values, **{f'{m}_status': s for m, s in status.items()}))
        print(f'Scored cells: {len(rows)}/{len(groups)}', flush=True)
    return pd.DataFrame(rows)


def plan_calls(config, bank):
    if config['k_repeats'] < 2 or config['l_paraphrases'] < 1:
        raise ValueError('All six metrics require K>=2 and L>=1')
    selected = set(config['selected_dialogue_ids'])
    if selected != {d['dialogue_id'] for d in bank['dialogues'] if d['dialogue_id'] in selected}:
        raise ValueError('Unknown dialogue selection')
    for model in config['models']:
        for dialogue in bank['dialogues']:
            if dialogue['dialogue_id'] not in selected:
                continue
            for condition in config['conditions']:
                prior = []
                for turn in dialogue['turns']:
                    variants = turn['paraphrases'][:config['l_paraphrases']]
                    if len(variants) != config['l_paraphrases'] or len(set(variants)) != len(variants):
                        raise ValueError('Insufficient distinct frozen paraphrases')
                    messages = build_primary_validation_messages(
                        turn=turn, condition=condition, prior_user_turns=prior,
                        system_prompt=config['system_prompt'])
                    requests = [('primary' if i == 0 else 'repeat', i, turn['user_prompt'])
                                for i in range(config['k_repeats'])]
                    requests += [('paraphrase', i, prompt) for i, prompt in enumerate(variants)]
                    for role, index, prompt in requests:
                        branch = [dict(x) for x in messages]
                        branch[-1]['content'] = prompt
                        yield dict(model=model['name'], dialogue_id=dialogue['dialogue_id'],
                                   condition=condition, turn=turn['turn'], qid=turn['qid'],
                                   template=dialogue['template'], role=role, call_index=index,
                                   gold_answer=turn['gold_answer'], messages=branch,
                                   messages_sha256=messages_sha256(branch), max_tokens=model['max_tokens'])
                    prior.append(turn['user_prompt'])


def call_key(job):
    return tuple(job[k] for k in ('model', 'dialogue_id', 'condition', 'turn', 'role', 'call_index'))


def execute_calls(jobs, clients, checkpoint, config_hash, callback):
    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)  # Corrupt checkpoints fail closed, never silently skipped.
            if record['config_hash'] != config_hash or call_key(record) in saved:
                raise ValueError('Mismatched config or duplicate checkpoint')
            saved[call_key(record)] = record
    expected = {call_key(j) for j in jobs}
    if set(saved) - expected:
        raise ValueError('Unexpected checkpoint records')
    with path.open('a', encoding='utf-8') as stream:
        for job in jobs:
            key = call_key(job)
            if key in saved:
                if saved[key]['messages_sha256'] != job['messages_sha256']:
                    raise ValueError('Changed request on resume')
                continue
            seed = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)
            sample = clients[job['model']].generate_messages(job['messages'], seed=seed, gold='')
            record = dict(job, config_hash=config_hash, call_seed=seed,
                          answer=sample.answer, raw_text=sample.raw_text,
                          reasoning_steps=list(sample.reasoning_steps), ok=sample.ok,
                          n_tokens=sample.n_tokens, latency_s=sample.latency_s,
                          finish_reason=sample.finish_reason)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            saved[key] = record
            callback(len(saved))
    return list(saved.values())
