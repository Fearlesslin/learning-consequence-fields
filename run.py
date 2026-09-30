from __future__ import annotations

import argparse

from src.protected_frontier.config import load_config
from src.protected_frontier.runner import check_package, run_experiments
from src.protected_frontier.summary import summarize


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run protected-frontier scheduling experiments.")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check packaged inputs")
    run = commands.add_parser("run", help="run replay experiments")
    run.add_argument("--root", type=int, action="append")
    run.add_argument("--load", type=float, action="append")
    run.add_argument("--method", action="append")
    run.add_argument("--save-details", action="store_true")
    commands.add_parser("summarize", help="aggregate saved metrics")
    all_command = commands.add_parser("all", help="run and summarize")
    all_command.add_argument("--save-details", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config()
    if args.command == "check":
        status = check_package(config)
        print(
            f"ok: {status['roots']} roots, {status['jobs']} jobs, "
            f"{status['models']} models"
        )
        return
    if args.command == "run":
        run_experiments(
            config,
            roots=args.root,
            loads=args.load,
            methods=args.method,
            save_details=args.save_details,
        )
        return
    if args.command == "summarize":
        summarize(config)
        return
    run_experiments(config, save_details=args.save_details)
    summarize(config)


if __name__ == "__main__":
    main()
