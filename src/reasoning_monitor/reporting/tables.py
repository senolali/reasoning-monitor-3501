"""LaTeX table generation.

Emits `booktabs` tables ready to \\input into a manuscript. Numbers are
formatted once, here, so a value cannot appear with different precision in the
table and the text. p-values below 0.001 are written as $<.001$ rather than
$0.000$, which would be false.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{digits}f}"


def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "$<.001$"
    return f"{p:.3f}".lstrip("0") if p < 1 else f"{p:.3f}"


def _escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _wrap(body: str, colspec: str, header: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def table_change_tests(df: pd.DataFrame, out: str | Path,
                       caption: str = "Pre- versus post-drift comparison of each "
                                      "reasoning dimension (Mann--Whitney $U$).",
                       label: str = "tab:change-tests") -> Path:
    header = (
        r"Metric & $\bar{x}_{\text{pre}}$ & $\bar{x}_{\text{post}}$ & $\Delta$ & "
        r"\% change & $U$ & $p$ & $r_{rb}$ \\"
    )
    rows = []
    for _, r in df.iterrows():
        sig = r"$^{*}$" if r.get("significant", False) else ""
        rows.append(
            f"{_escape(r['metric'])}{sig} & {_fmt(r['mean_before'])} & "
            f"{_fmt(r['mean_after'])} & {_fmt(r['delta'])} & "
            f"{_fmt(r['pct_change'], 1)} & {_fmt(r['u_statistic'], 1)} & "
            f"{_fmt_p(r['p_value'])} & {_fmt(r['rank_biserial'])} \\\\"
        )
    body = "\n".join(rows)
    tex = _wrap(body, "lrrrrrrr", header, caption, label)
    tex += (
        "\n% $^{*}$ significant after Holm--Bonferroni correction at "
        "$\\alpha=.05$.\n"
    )
    return _write(tex, out)


def table_detection(df: pd.DataFrame, out: str | Path,
                    caption: str = "Detection performance per metric--detector pair.",
                    label: str = "tab:detection") -> Path:
    header = r"Metric & Detector & Detected & Delay & False alarms & FA rate (\%) \\"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{_escape(r['metric'])} & {_escape(r['detector'])} & "
            f"{'yes' if r['detected'] else 'no'} & {_fmt(r['delay'], 1)} & "
            f"{int(r['n_false_alarms'])} & {_fmt(r['false_alarm_rate'], 2)} \\\\"
        )
    return _write(_wrap("\n".join(rows), "llccrr", header, caption, label), out)


def table_early_warning(df: pd.DataFrame, out: str | Path,
                        caption: str = "Early-warning ranking: mean detection delay "
                                       "across drift scenarios.",
                        label: str = "tab:early-warning") -> Path:
    header = (r"Rank & Metric & Detection rate & Mean delay & Median delay & "
              r"Mean FA rate (\%) \\")
    rows = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        rows.append(
            f"{i} & {_escape(r['metric'])} & {_fmt(r['detection_rate'], 2)} & "
            f"{_fmt(r['mean_delay'], 1)} & {_fmt(r['median_delay'], 1)} & "
            f"{_fmt(r['mean_false_alarm_rate'], 2)} \\\\"
        )
    return _write(_wrap("\n".join(rows), "rlrrrr", header, caption, label), out)


def table_assignment(rows: list[dict], out: str | Path,
                     caption: str = "Metric--detector assignment by strategy.",
                     label: str = "tab:assignment") -> Path:
    header = r"Metric & Detector & Strategy \\"
    body = "\n".join(
        f"{_escape(r['metric'])} & {_escape(r['detector'])} & "
        f"{_escape(r['strategy'])} \\\\"
        for r in rows
    )
    return _write(_wrap(body, "lll", header, caption, label), out)


def table_scenario_summary(df: pd.DataFrame, out: str | Path,
                           caption: str = "Scenario definitions and observed "
                                          "composite-quality change.",
                           label: str = "tab:scenarios") -> Path:
    header = (r"Scenario & Drift type & $t^{*}$ & $N$ & $Q$ pre & $Q$ post & "
              r"$\Delta Q$ \\")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{_escape(r['scenario'])} & {_escape(r['drift_type'])} & "
            f"{int(r['transition_start']) if r.get('transition_start', -1) > 0 else '--'} & "
            f"{int(r['n_queries'])} & {_fmt(r['q_pre'])} & {_fmt(r['q_post'])} & "
            f"{_fmt(r['q_delta'])} \\\\"
        )
    return _write(_wrap("\n".join(rows), "llrrrrr", header, caption, label), out)


def _write(tex: str, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out
