"""Call-free content diagnostics with temporal figures and next-turn evaluation."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reasoning_monitor.dialogue_experiment import records_to_turn_frame  # noqa: E402
from reasoning_monitor.evaluation.dialogue_content import content_frame  # noqa: E402
from reasoning_monitor.evaluation.dialogue_warning import (  # noqa: E402
    grouped_predictions,
    prediction_rows,
    score_predictions,
)

HASHES = {
    'gpt4o_mini': 'fb21699a1106b3414f2df1f62741eb1fade59751c6e69f98d8656974d3bc6529',
    'llama4_scout': '2a00f004690c63881cb9b670191514fcf54214216388ec35d18c91c1e9eb68c2',
}
BASE = ['turn', 'context_chars']
SETS = {
    'turn_context': BASE,
    'arithmetic': BASE + ['arithmetic_available', 'arithmetic_error_rate', 'equation_count'],
    'transition': BASE + ['transition_available', 'transition_violation'],
    'combined': BASE + ['arithmetic_available', 'arithmetic_error_rate', 'equation_count',
                        'transition_available', 'transition_violation'],
}


def figures(frames, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    heat, heat_axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for column, (model, frame) in enumerate(frames.items()):
        title = {'gpt4o_mini': 'GPT-4o-mini', 'llama4_scout': 'Llama-4 Scout'}[model]
        for condition, label, colour in [('user_only_history', 'Ham gecmis', '#0072B2'),
                                         ('state_summary', 'Dogrulanmis ozet', '#009E73')]:
            part = frame[frame.condition == condition]
            mean = part.groupby('turn').CQ_raw.mean()
            axes[0, column].plot(mean.index, mean, 'o-', color=colour, label=label)
        user = frame[frame.condition == 'user_only_history']
        rates = user.groupby('turn')[['arithmetic_any_error', 'transition_violation',
                                      'arithmetic_available']].mean()
        for metric, label, colour in [('arithmetic_any_error', 'Hesap hatasi', '#D55E00'),
                                     ('transition_violation', 'Guncelleme tutarsizligi', '#CC79A7'),
                                     ('arithmetic_available', 'Hesap denetimi kapsami', '#777777')]:
            series = rates[metric].copy()
            if metric == 'transition_violation':
                series.loc[1] = float('nan')
            axes[1, column].plot(series.index, series, 'o-', color=colour, label=label)
        tokens = user.groupby('turn').n_tokens.mean()
        axes[2, column].plot(tokens.index, tokens, 'o-', color='#0072B2')
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel('Dogru yanit orani (9 diyalog)')
        axes[1, column].set_ylabel('Isaretlenen / toplam diyalog')
        axes[2, column].set_ylabel('Ortalama yanit token sayisi')
        axes[2, column].set_xlabel('Diyalog turu')
        for row in (0, 1):
            axes[row, column].set_ylim(-0.05, 1.05)
            axes[row, column].legend(fontsize=8, loc='lower left')
        for row in range(3):
            axes[row, column].set_xticks(range(1, 13))
            axes[row, column].grid(alpha=0.2)
        pivot = user.pivot(index='dialogue_id', columns='turn', values='CQ_raw')
        ax = heat_axes[column]
        ax.imshow(pivot, cmap=matplotlib.colors.ListedColormap(['#f6c5a8', '#b9dfd1']),
                  vmin=0, vmax=1, aspect='auto')
        for y, dialogue in enumerate(pivot.index):
            part = user[user.dialogue_id == dialogue]
            for _, cell in part.iterrows():
                x = int(cell.turn) - 1
                if cell.transition_violation:
                    ax.scatter(x, y, marker='o', s=110, facecolors='none', edgecolors='#762a83', linewidth=1.8)
                if cell.arithmetic_any_error:
                    ax.scatter(x, y, marker='x', s=50, color='#222222')
            failures = part[part.CQ_raw == 0]
            if len(failures):
                ax.scatter(int(failures.turn.min())-1, y, marker='s', s=210,
                           facecolors='none', edgecolors='#222222', linewidth=1.2)
        ax.set_yticks(range(9), [x.replace('cdr-v14-', '') for x in pivot.index])
        ax.set_xticks(range(12), range(1, 13))
        ax.set_title(title + ' | Yesil: dogru, turuncu: yanlis; kare: ilk hata')
        ax.set_xlabel('Tur | Mor halka: guncelleme tutarsizligi; x: acik hesap hatasi')
    fig.suptitle('CDR-v1.4 | Tur boyunca degisim (kesifsel, model basina 9 diyalog)')
    heat.suptitle('Hata ve icerik sinyallerinin zamani | Ham gecmis kosulu')
    for figure, name in [(fig, 'trajectories'), (heat, 'event_timeline')]:
        figure.savefig(output / f'{name}.png', dpi=170)
        figure.savefig(output / f'{name}.pdf')
        plt.close(figure)


def main():
    output = ROOT / 'results/controlled_dialogue_v1_4_content_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(no_model_calls=True, exploratory=True, original_gate_unchanged=True,
                  feature_sets=SETS, source_sha256=HASHES, models={}, limitations=[
                      'Numeric equalities only; unsupported expressions are not judged correct.',
                      'Transition checks are template-specific and use previous model answer, not gold.',
                      'Transition violation may be error correction, not necessarily a new error.',
                      'For exact additive tasks, consistent correct answers cannot violate transitions before first error.',
                      'Same-turn detection is not first-error early warning.',
                      'Three template families; no significance or causal claims.',
                  ])
    frames = {}
    for model, digest in HASHES.items():
        path = ROOT / f'results/controlled_dialogue_v1_4_replication/data/records_CDR14_{model}.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('source changed')
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        content, audit = content_frame(records)
        full = records_to_turn_frame(records).merge(content, on=['qid', 'condition'], validate='one_to_one')
        frames[model] = full
        full.to_csv(output / f'metrics_{model}.csv', index=False)
        (output / f'equations_{model}.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        frame = prediction_rows(records).merge(content, on=['qid', 'condition'], validate='one_to_one')
        result = {}
        for group in ('dialogue_id', 'template'):
            result[group] = {}
            for name, features in SETS.items():
                probability = grouped_predictions(frame, features, group)
                frame[f'p_{group}_{name}'] = probability
                result[group][name] = score_predictions(frame, probability)
        user = full[full.condition == 'user_only_history']
        first = frame[frame.before_first_error & (frame.next_error == 1)]
        result['coverage'] = dict(
            arithmetic_responses=int(user.arithmetic_available.sum()), total_responses=len(user),
            explicit_equations=int(user.equation_count.sum()),
            transition_responses=int(user.transition_available.sum()),
            arithmetic_flagged=int(user.arithmetic_any_error.sum()),
            transition_flagged=int(user.transition_violation.sum()),
            direct_arithmetic_flags_before_first_error=int(first.arithmetic_any_error.sum()),
            direct_transition_flags_before_first_error=int(first.transition_violation.sum()),
            first_error_opportunities=len(first),
        )
        frame.to_csv(output / f'predictions_{model}.csv', index=False)
        report['models'][model] = result
        print(model, json.dumps(result['coverage']), flush=True)
    figures(frames, output)
    report['code_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ['scripts/analyse_dialogue_content.py',
                     'src/reasoning_monitor/evaluation/dialogue_content.py',
                     'src/reasoning_monitor/evaluation/dialogue_warning.py']
    }
    (output / 'content_assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for model, result in report['models'].items():
        print(model, 'leave-template-out Brier:', {n: round(s['brier'], 5) for n, s in result['template'].items()})


if __name__ == '__main__':
    main()
