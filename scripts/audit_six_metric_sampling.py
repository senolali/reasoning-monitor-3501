"""Read the existing probe and score locally. Never creates remote clients."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dialogue_six_metrics import digest, load_plan  # noqa: E402

from reasoning_monitor.dialogue_six_metrics import METRICS, StrictScorers  # noqa: E402
from reasoning_monitor.evaluation.sampling_audit import audit_probe  # noqa: E402


def plot(report, output):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for metric, ax in zip(METRICS, axes.flat, strict=True):
        ax.set_title(metric + (' (primary only)' if metric in ('CQ', 'LS') else ''))
        for model, color, offset in [('gpt4o_mini', '#0072B2', -0.08),
                                     ('llama4_scout', '#D55E00', 0.08)]:
            cells = [c for c in report['cells'] if c['model'] == model]
            xs = [c['turn'] + offset for c in cells if c['scores'][metric] is not None]
            ys = [c['scores'][metric] for c in cells if c['scores'][metric] is not None]
            if xs:
                ax.scatter(xs, ys, label=model, color=color, s=45)
                ax.legend(loc='lower left', fontsize=8)
        if metric in ('RS', 'ES'):
            ax.text(0.5, 0.5, 'Not measured\n' + ('No paraphrases' if metric == 'RS'
                    else 'No consecutive window'), ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([2, 11])
        ax.set_xlim(1, 12)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlabel('Turn (only 2 and 11 sampled)')
        ax.set_ylabel('Score')
        ax.grid(alpha=0.2)
    fig.suptitle('Offline sampling audit | 1 dialogue, 5 samples/cell, 20 calls total\n'
                 'Isolated observations, not a continuous trajectory or early-warning test')
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'probe_metrics.{suffix}', dpi=150)
    plt.close(fig)


def main():
    config, jobs, fingerprint = load_plan()
    source = ROOT / config['output_dir'] / 'sampling_probe.jsonl'
    old_report = source.with_name('sampling_probe_report.json')
    originals = {str(p.relative_to(ROOT)): digest(p) for p in (source, old_report)}
    expected = [j for j in jobs if j['dialogue_id'] == config['selected_dialogue_ids'][0]
                and j['condition'] == 'user_only_history' and j['turn'] in (2, 11)
                and j['role'] != 'paraphrase']
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    print('Offline: loading cached scorers; no remote model calls.', flush=True)
    scorer = StrictScorers(with_ss=True)
    report = audit_probe(records, expected, fingerprint, scorer, config['k_repeats'])
    report.update(source_sha256=originals, config_hash=fingerprint,
                  original_gate_pass=json.loads(old_report.read_text(encoding='utf-8'))['passed'],
                  scorer=dict(nli='cross-encoder/nli-deberta-v3-small',
                              bertscore='distilbert-base-uncased', rescale_with_baseline=False))
    output = ROOT / 'results/dialogue_six_metrics_sampling_audit'
    output.mkdir(parents=True, exist_ok=True)
    plot(report, output)
    for path, sha in originals.items():
        if digest(ROOT / path) != sha:
            raise ValueError('Original evidence changed during audit')
    (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
