from __future__ import annotations

import argparse
import json

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio x-pred preference Stage 2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "train",
        "collect-dpo",
        "prepare-passt-cache",
        "prepare-fad-cache",
        "prepare-constraint-caches",
        "describe",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        if command in {"train", "collect-dpo"}:
            child.add_argument("--output-dir")
        if command == "train":
            child.add_argument("--preference-dir")
            child.add_argument("--resume-from")
        if command in {
            "prepare-passt-cache",
            "prepare-fad-cache",
            "prepare-constraint-caches",
        }:
            child.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "describe":
        print(json.dumps(config.to_dict(), indent=2))
        return
    if args.command == "prepare-passt-cache":
        from .prepare_passt_cache import prepare_passt_cache

        prepare_passt_cache(config, batch_size=args.batch_size)
        return
    if args.command == "prepare-fad-cache":
        from .prepare_fad_cache import prepare_fad_cache

        prepare_fad_cache(config, batch_size=args.batch_size)
        return
    if args.command == "prepare-constraint-caches":
        from .prepare_constraint_caches import prepare_constraint_caches

        prepare_constraint_caches(config, batch_size=args.batch_size)
        return
    if args.command == "collect-dpo":
        from .flow_dpo import collect_preferences

        collect_preferences(config, output_dir=args.output_dir)
        return
    if config.algorithm == "flow_grpo":
        from .flow_grpo import train_flow_grpo

        train_flow_grpo(
            config, output_dir=args.output_dir, resume_from=args.resume_from
        )
    else:
        if not args.preference_dir:
            raise SystemExit("flow_dpo training requires --preference-dir")
        from .flow_dpo import train_flow_dpo

        train_flow_dpo(
            config,
            preference_dir=args.preference_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
