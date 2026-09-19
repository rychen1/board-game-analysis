"""CLI: descriptive analysis of a local Game JSONL corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from board_game_analysis.analysis.run import (
    run_corpus_descriptive,
    run_corpus_robustness,
)
from board_game_analysis.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bga-analyze",
        description=(
            "Descriptive analysis of a local BGG Game JSONL. Default corpus "
            "is bgg_boardgames_v1. Not a population study."
        ),
    )
    parser.add_argument(
        "--corpus-id",
        default="bgg_boardgames_v1",
        help="Packaged corpus id (default: bgg_boardgames_v1)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Override the default processed JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the default analysis output directory",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help=(
            "Run the second-pass robustness stress test instead of the "
            "first-pass descriptive analysis"
        ),
    )
    args = parser.parse_args(argv)
    settings = Settings()
    try:
        if args.robustness:
            result = run_corpus_robustness(
                settings,
                corpus_id=args.corpus_id,
                jsonl_path=args.jsonl,
                output_dir=args.output_dir,
            )
        else:
            result = run_corpus_descriptive(
                settings,
                corpus_id=args.corpus_id,
                jsonl_path=args.jsonl,
                output_dir=args.output_dir,
            )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"report\t{result.report_path}")
    print(f"findings\t{result.findings_path}")
    print(f"manifest\t{result.manifest_path}")
    print(f"corpus_payload_id\t{result.corpus_payload_id}")
    print(f"report_payload_id\t{result.report_payload_id}")
    print(f"record_id\t{result.record_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
