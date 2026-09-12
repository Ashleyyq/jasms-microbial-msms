#!/usr/bin/env python3
"""Verify that Carafe sensitivity runs changed only seed and output paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path


SEEDS = (2024, 2025, 2026)
ARMS = ("C1", "C2", "C3")
RUNS = ("P2", "P3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_command(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"missing command file: {path}")
    return shlex.split(path.read_text())


def option_value(tokens: list[str], option: str) -> str:
    matches = [index for index, token in enumerate(tokens) if token == option]
    if len(matches) != 1 or matches[0] + 1 >= len(tokens):
        raise SystemExit(f"expected one {option} value in command")
    return tokens[matches[0] + 1]


def normalize(tokens: list[str], variable_options: tuple[str, ...]) -> list[str]:
    result = list(tokens)
    for option in variable_options:
        matches = [index for index, token in enumerate(result) if token == option]
        if len(matches) != 1 or matches[0] + 1 >= len(result):
            raise SystemExit(f"expected one {option} value in command")
        result[matches[0] + 1] = f"<{option}>"
    return result


def training_path(root: Path, seed: int, arm: str) -> Path:
    if seed == 2024:
        return root / "carafe" / arm / "command.txt"
    return root / "robustness" / f"carafe_seed_{seed}" / "carafe" / arm / "command.txt"


def search_path(root: Path, seed: int, arm: str, run: str) -> Path:
    if seed == 2024:
        return root / "carafe_search" / arm / f"test_{run}" / "command.txt"
    return (
        root
        / "robustness"
        / f"carafe_seed_{seed}"
        / "carafe_search"
        / arm
        / f"test_{run}"
        / "command.txt"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result: dict[str, object] = {
        "status": "PASS",
        "seeds": list(SEEDS),
        "training": {},
        "search": {},
    }
    for arm in ARMS:
        commands = {seed: read_command(training_path(args.run_root, seed, arm)) for seed in SEEDS}
        observed_seeds = {seed: int(option_value(command, "-seed")) for seed, command in commands.items()}
        if observed_seeds != {seed: seed for seed in SEEDS}:
            raise SystemExit(f"unexpected random seeds for {arm}: {observed_seeds}")
        normalized = {
            seed: normalize(command, ("-o", "-seed"))
            for seed, command in commands.items()
        }
        if len({tuple(command) for command in normalized.values()}) != 1:
            raise SystemExit(f"training commands differ beyond output and seed for {arm}")
        result["training"][arm] = {
            "observed_random_seeds": observed_seeds,
            "only_differences": ["-o", "-seed"],
            "command_sha256": {
                seed: sha256(training_path(args.run_root, seed, arm)) for seed in SEEDS
            },
        }

    for arm in ARMS:
        result["search"][arm] = {}
        for run in RUNS:
            commands = {
                seed: read_command(search_path(args.run_root, seed, arm, run))
                for seed in SEEDS
            }
            normalized = {
                seed: normalize(command, ("--lib", "--temp", "--out"))
                for seed, command in commands.items()
            }
            if len({tuple(command) for command in normalized.values()}) != 1:
                raise SystemExit(
                    f"search commands differ beyond library/output paths for {arm} {run}"
                )
            result["search"][arm][run] = {
                "only_differences": ["--lib", "--temp", "--out"],
                "command_sha256": {
                    seed: sha256(search_path(args.run_root, seed, arm, run))
                    for seed in SEEDS
                },
            }

    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
