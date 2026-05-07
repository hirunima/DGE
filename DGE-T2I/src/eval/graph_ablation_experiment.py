"""CLI entrypoint for graph-grounded ablation experiments."""

from __future__ import annotations

from .ablation import build_arg_parser, config_from_args, run_ablation_experiment, write_experiment_outputs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    report = run_ablation_experiment(config)
    write_experiment_outputs(report, config.output_dir)


if __name__ == "__main__":
    main()
