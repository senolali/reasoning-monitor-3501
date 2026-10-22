"""Plan, preflight, collect and plot all six dialogue metrics; default is call-free."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reasoning_monitor.dialogue_six_metrics import (  # noqa: E402
    METRICS,
    StrictScorers,
    call_key,
    execute_calls,
    plan_calls,
    score_records,
)
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_metrics(frame, output):
    panels = list(frame.groupby('model'))
    panels += [(f'{model}_{dialogue}', data) for (model, dialogue), data in
               frame.groupby(['model', 'dialogue_id'])]
    for model, data in panels:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
        for metric, ax in zip(METRICS, axes.flat, strict=True):
            ax.set_title(metric + (' (gold-dependent)' if metric in ('CQ', 'RS', 'ES') else ''))
            if metric == 'ES':
                ax.axvspan(1, 4.5, color='#aaaaaa', alpha=0.12)
                ax.text(0.02, 0.92, '1-4: pencere isinmasi', transform=ax.transAxes, fontsize=8)
            if data[metric].notna().any():
                for condition, label, color in [('user_only_history', 'Ham gecmis', '#0072B2'),
                                                 ('state_summary', 'Dogrulanmis ozet', '#009E73')]:
                    subset = data[data.condition == condition]
                    group = subset.groupby('turn')[metric]
                    mean, count = group.mean(), group.count()
                    ax.plot(mean.index, mean, 'o-', label=label, color=color)
                    ax.text(0.02, 0.07 if condition == 'user_only_history' else 0.01,
                            f'{label}: n/tur={count.min()}..{count.max()}', transform=ax.transAxes,
                            fontsize=8, color=color)
                ax.legend(loc='upper right', fontsize=8)
            else:
                reason = 'K tekrar yok' if metric in ('CS', 'SS') else 'Yeniden ifade yok'
                if metric == 'LS':
                    reason = 'NLI kullanilamiyor'
                ax.text(0.5, 0.5, 'OLCULMEDI\n' + reason, transform=ax.transAxes,
                        ha='center', va='center', color='#777777')
            ax.set_ylim(-0.03, 1.15)
            ax.set_xticks(range(1, 13))
            ax.grid(alpha=0.2)
            ax.set_xlabel('Diyalog turu')
            ax.set_ylabel('Skor (yuksek = daha iyi)')
        fig.suptitle(f'{model} | Alti metrik | Kesifsel kalibrasyon\n'
                     'ES: son 5 tur token araligi; RS: yalniz dogru birincil yanitlar')
        for extension in ('png', 'pdf'):
            fig.savefig(output / f'six_metrics_{model}.{extension}', dpi=150)
        plt.close(fig)


def load_plan():
    path = ROOT / 'configs/dialogue_six_metrics.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    base_path = ROOT / config['base_config']
    if digest(base_path) != config['base_sha256']:
        raise ValueError('Base config changed')
    base = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    base.update(config)
    for model in base['models']:
        model['temperature'] = config['temperature']
    bank_path = ROOT / base['bank_path']
    if digest(bank_path) != base['bank_file_sha256']:
        raise ValueError('Bank changed')
    bank = json.loads(bank_path.read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    return base, list(plan_calls(base, bank)), fingerprint


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--historical', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--probe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    config, jobs, fingerprint = load_plan()
    output = ROOT / config['output_dir']
    probe_path = output / 'sampling_probe.jsonl'
    full_path = output / 'records.jsonl'
    print(json.dumps(dict(calls=len(jobs), sampling_probe_calls=20,
                          maximum_output_tokens=sum(j['max_tokens'] for j in jobs),
                          k=config['k_repeats'], l=config['l_paraphrases'],
                          temperature=config['temperature'], config_hash=fingerprint)), flush=True)
    if not any(vars(args).values()):
        return
    if args.historical:
        output = ROOT / 'results/controlled_dialogue_v1_4_six_metric_rescore'
        records = []
        expected = {'gpt4o_mini': 'fb21699a1106b3414f2df1f62741eb1fade59751c6e69f98d8656974d3bc6529',
                    'llama4_scout': '2a00f004690c63881cb9b670191514fcf54214216388ec35d18c91c1e9eb68c2'}
        for model, sha in expected.items():
            path = ROOT / f'results/controlled_dialogue_v1_4_replication/data/records_CDR14_{model}.jsonl'
            if digest(path) != sha:
                raise ValueError('Historical source changed')
            records.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
        scorer = StrictScorers(with_ss=False)
    else:
        print('Loading real local NLI and BERTScore scorers...', flush=True)
        scorer = StrictScorers(with_ss=True)  # Fail before any paid call if scorer unavailable.
        if args.preflight:
            print('Scorers ready; network model sampling still needs the probe.')
            return
        if args.probe:
            first_id = config['selected_dialogue_ids'][0]
            jobs = [j for j in jobs if j['dialogue_id'] == first_id
                    and j['condition'] == 'user_only_history' and j['turn'] in (2, 11)
                    and j['role'] != 'paraphrase']
            target = probe_path
        else:
            target = full_path
        if args.execute:
            if not probe_path.exists():
                raise ValueError('Run and inspect sampling probe before full execution')
            probe = [json.loads(x) for x in probe_path.read_text(encoding='utf-8').splitlines()]
            for model in config['models']:
                for turn in (2, 11):
                    cell = [r for r in probe if r['model'] == model['name'] and r['turn'] == turn]
                    if len(cell) != config['k_repeats'] or any(
                        not r['ok'] or r['config_hash'] != fingerprint
                        or r.get('finish_reason') == 'length' or r['n_tokens'] >= r['max_tokens']
                        for r in cell
                    ) or len({r['raw_text'] for r in cell}) < 2:
                        raise ValueError(f'Sampling capability not established: {model["name"]} turn {turn}')
        if args.score_only:
            records = [json.loads(x) for x in target.read_text(encoding='utf-8').splitlines()]
            expected = {call_key(j): j['messages_sha256'] for j in jobs}
            if len(records) != len(expected) or {call_key(r) for r in records} != set(expected):
                raise ValueError('Incomplete or duplicate records: resume acquisition first')
            if any(r['config_hash'] != fingerprint or r['messages_sha256'] != expected[call_key(r)]
                   for r in records):
                raise ValueError('Scoring source does not match frozen plan')
        else:
            completed = len(target.read_text(encoding='utf-8').splitlines()) if target.exists() else 0
            reporter = ProgressReporter(total=len(jobs), label='Six-metric calls', completed=completed)
            clients = {m['name']: build_client(m) for m in config['models']}
            reporter.print(completed)
            records = execute_calls(jobs, clients, target, fingerprint, reporter.print)
            reporter.print(len(records), final=True)
        if args.probe:
            cells = []
            for model in config['models']:
                for turn in (2, 11):
                    cell = [r for r in records if r['model'] == model['name'] and r['turn'] == turn]
                    distinct = len({r['raw_text'] for r in cell})
                    cells.append(dict(model=model['name'], turn=turn, distinct_responses=distinct,
                                      passed=len(cell) == config['k_repeats'] and distinct >= 2
                                      and all(r['ok'] and r.get('finish_reason') != 'length'
                                              and r['n_tokens'] < r['max_tokens'] for r in cell)))
            report = dict(cells=cells, passed=all(c['passed'] for c in cells),
                          note='Identical responses do not prove ignored sampling parameters.')
            (output / 'sampling_probe_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2))
            return
    output.mkdir(parents=True, exist_ok=True)
    frame = score_records(records, scorer, config['k_repeats'], config['l_paraphrases'])
    frame.to_csv(output / 'six_metrics.csv', index=False)
    plot_metrics(frame, output)
    status = {m: frame[f'{m}_status'].value_counts().to_dict() for m in METRICS}
    manifest = dict(status=status, historical=args.historical, no_model_calls=args.historical or args.score_only,
                    nli_model='cross-encoder/nli-deberta-v3-small',
                    nli_revision=getattr(scorer.nli.model.config, '_commit_hash', None),
                    bertscore_model='distilbert-base-uncased' if scorer.bert else None,
                    source_records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                    config_hash=fingerprint, exploratory=True,
                    warning='RS excludes incorrect primaries; ES depends on CQ. Missing scores are not zeros.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
