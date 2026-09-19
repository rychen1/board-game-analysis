"""CLI: ingest explicit BGG thing ids or a frozen corpus list."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from board_game_analysis.config import Settings
from board_game_analysis.ingestion.errors import IngestionError
from board_game_analysis.ingestion.pipeline import ingest_bgg_games, ingest_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="board-game-ingest",
        description=(
            "Ingest BoardGameGeek XML API2 thing records. Pass thing ids, or "
            "use --corpus for the frozen small-corpus list (~50 ids; designed "
            "to scale to ~1000)."
        ),
    )
    parser.add_argument(
        "source_ids",
        nargs="*",
        help="BGG thing ids (for example 13 for CATAN)",
    )
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="Ingest the frozen packaged id list (resume from raw cache)",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Optional id list (comments allowed). Default: packaged corpus.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest at most this many ids from the corpus list (smoke: 25–50)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many ids from the start of the corpus list",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch from BGG even if a raw archive file exists",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        print("error: --limit must be >= 0", file=sys.stderr)
        return 2
    if args.offset < 0:
        print("error: --offset must be >= 0", file=sys.stderr)
        return 2
    if not args.corpus and not args.source_ids:
        parser.error("provide thing ids or pass --corpus")
    if args.corpus and args.source_ids:
        parser.error("pass thing ids or --corpus, not both")
    extra_corpus_flags = (
        args.ids_file is not None or args.limit is not None or args.offset
    )
    if not args.corpus and extra_corpus_flags:
        parser.error("--ids-file, --limit, and --offset require --corpus")

    settings = Settings()
    try:
        if args.corpus:
            return _run_corpus(args, settings)
        return _run_ids(args, settings)
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_ids(args: argparse.Namespace, settings: Settings) -> int:
    games = ingest_bgg_games(args.source_ids, settings, force=args.force)
    for game in games:
        identifier = game.sources[0].source_identifier if game.sources else game.id
        print(f"{identifier}\t{game.title}\t{game.id}")
    return 0


def _run_corpus(args: argparse.Namespace, settings: Settings) -> int:
    result = ingest_corpus(
        settings,
        ids_file=args.ids_file,
        offset=args.offset,
        limit=args.limit,
        force=args.force,
    )
    counts = {
        "requested": len(result.requested_ids),
        "ok": len(result.ok),
        "skipped": len(result.skipped),
        "errors": len(result.errors),
    }
    print(
        f"ingested {counts['ok']}/{counts['requested']}  "
        f"skipped={counts['skipped']}  errors={counts['errors']}"
    )
    print(f"manifest\t{result.manifest_path}")
    print(f"jsonl\t{result.jsonl_path}")
    for record in result.errors:
        print(f"error\t{record.source_id}\t{record.reason}", file=sys.stderr)
    for record in result.skipped:
        print(f"skipped\t{record.source_id}\t{record.reason}\t{record.item_type}")
    if counts["requested"] > 0 and counts["ok"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
