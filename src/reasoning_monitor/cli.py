"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reasoning_monitor.experiment import ExperimentConfig, run_experiment


def apply_quick_preset(cfg: ExperimentConfig) -> ExperimentConfig:
    """Small, fast, offline smoke run — also what CI executes.

    ST2 is the composition shift and ST3 a stationary control, so the preset
    exercises both the detection and the false-alarm path.  It is a function so
    a test can assert the scenario names still exist: the previous preset asked
    for "S1"/"S2", names retired in v0.2.0, and every `--quick` run (CI included)
    died with `KeyError: 'S1'`.
    """
    cfg.items_per_dataset = 60
    cfg.scenarios = ["ST2", "ST3"]
    cfg.models = cfg.models[:2]
    cfg.seeds = [cfg.seed]
    cfg.run_ablation = False
    cfg.run_strategy_comparison = False
    cfg.run_replication = False
    cfg.use_rqeval = False
    cfg.coherence_scorer = "lexical"
    cfg.stability_scorer = "jaccard"
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="reasoning-monitor",
        description="Run the reasoning-drift monitoring experiment end to end.",
    )
    p.add_argument("--config", type=str, help="YAML config file")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--items", type=int, default=300, help="items per dataset")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--window", type=int, nargs="+", default=None)
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--coherence", choices=["lexical", "nli", "auto"], default="lexical")
    p.add_argument("--stability", choices=["jaccard", "bertscore", "auto"], default="auto")
    p.add_argument("--strategy",
                   choices=["distribution_based", "proposal_mapping", "adwin_all"],
                   default="distribution_based")
    p.add_argument("--publication-mode", action="store_true",
                   help="require real NLI/BERTScore scorers and refuse a run whose "
                        "metrics cannot vary; makes preflight failures fatal")
    p.add_argument("--quick", action="store_true",
                   help="small, fast run for smoke-testing the pipeline")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose or True else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config:
        import yaml

        raw = yaml.safe_load(Path(args.config).read_text())
        cfg = ExperimentConfig(**raw)
    else:
        cfg = ExperimentConfig(
            seed=args.seed,
            output_dir=args.output_dir,
            items_per_dataset=args.items,
            coherence_scorer=args.coherence,
            stability_scorer=args.stability,
            detector_strategy=args.strategy,
            publication_mode=args.publication_mode,
            run_ablation=not args.no_ablation,
        )
        if args.scenarios:
            cfg.scenarios = args.scenarios
        if args.window:
            cfg.window_sizes = args.window

    if args.quick:
        apply_quick_preset(cfg)

    manifest = run_experiment(cfg)
    print(f"\nResults written to {Path(cfg.output_dir).resolve()}")
    print(f"  figures/  {len(list((Path(cfg.output_dir) / 'figures').glob('*.png')))} PNG + PDF")
    print(f"  tables/   {len(list((Path(cfg.output_dir) / 'tables').glob('*.tex')))} LaTeX tables")
    print(f"  runtime   {manifest['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
