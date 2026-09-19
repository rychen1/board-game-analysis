"""Run the corpus-descriptive analysis against a local Game JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ds_platform import CodeRef, payload_id

from board_game_analysis.analysis.artifacts import (
    default_robustness_schema_path,
    report_json_bytes,
    store_descriptive_report,
)
from board_game_analysis.analysis.descriptive import (
    ANALYSIS_ID,
    ANALYSIS_LOGICAL_KEY,
    ANALYSIS_VERSION,
    build_rows,
    build_tables,
)
from board_game_analysis.analysis.plots import write_figures
from board_game_analysis.analysis.report import build_report, render_findings
from board_game_analysis.analysis.robustness import (
    ROBUSTNESS_ID,
    ROBUSTNESS_LOGICAL_KEY,
    ROBUSTNESS_VERSION,
    build_robustness_tables,
)
from board_game_analysis.analysis.robustness_plots import write_robustness_figures
from board_game_analysis.analysis.robustness_report import (
    build_robustness_report,
    render_robustness_findings,
)
from board_game_analysis.config import Settings
from board_game_analysis.ingestion.corpus.ids import corpus_spec
from board_game_analysis.ingestion.quality import load_games_jsonl, load_manifest
from board_game_analysis.ingestion.raw_artifacts import artifact_store, bga_run_context
from board_game_analysis.ingestion.util import try_git_code_ref


@dataclass
class AnalysisRunResult:
    report_path: Path
    findings_path: Path
    manifest_path: Path
    figure_paths: list[Path]
    corpus_payload_id: str
    report_payload_id: str
    record_id: str
    run_id: str


@dataclass
class RobustnessRunResult:
    report_path: Path
    findings_path: Path
    manifest_path: Path
    figure_paths: list[Path]
    corpus_payload_id: str
    report_payload_id: str
    record_id: str
    run_id: str


def default_jsonl_path(data_dir: Path, corpus_id: str) -> Path:
    spec = corpus_spec(corpus_id)
    return data_dir / "processed" / "boardgamegeek" / spec.jsonl_name


def default_output_dir(data_dir: Path) -> Path:
    return data_dir / "derived" / "analysis" / ANALYSIS_ID


def default_robustness_output_dir(data_dir: Path) -> Path:
    return data_dir / "derived" / "analysis" / ROBUSTNESS_ID


def run_corpus_descriptive(
    settings: Settings,
    *,
    corpus_id: str = "bgg_boardgames_v1",
    jsonl_path: Path | None = None,
    output_dir: Path | None = None,
) -> AnalysisRunResult:
    spec = corpus_spec(corpus_id)
    source = (
        jsonl_path
        if jsonl_path is not None
        else default_jsonl_path(settings.data_dir, corpus_id)
    )
    if not source.is_file():
        msg = (
            f"corpus JSONL not found: {source}. Ingest first with "
            f"`board-game-ingest --corpus --corpus-id {corpus_id}`."
        )
        raise FileNotFoundError(msg)
    started_at = datetime.now(UTC)
    loaded = load_games_jsonl(source)
    jsonl_bytes = source.read_bytes()
    corpus_pid = payload_id(jsonl_bytes)
    tables = build_tables(loaded.games)
    report = build_report(
        loaded.games,
        tables,
        corpus_id=spec.corpus_id,
        jsonl_name=source.name,
        corpus_payload_id=corpus_pid,
        started_at=started_at,
        load_errors=len(loaded.errors),
        ingest_counts=_ingest_counts(settings.data_dir, spec.corpus_id),
    )
    report_bytes = report_json_bytes(report)
    findings = render_findings(report)

    dest = (
        output_dir if output_dir is not None else default_output_dir(settings.data_dir)
    )
    dest.mkdir(parents=True, exist_ok=True)
    figures_dir = dest / "figures"
    figure_paths = write_figures(build_rows(loaded.games), tables, figures_dir)

    run = bga_run_context(started_at=started_at)
    code_ref = try_git_code_ref()
    if code_ref is not None:
        run = run.model_copy(update={"code_ref": CodeRef.model_validate(code_ref)})
    store = artifact_store(settings.data_dir)
    store.put(corpus_pid, jsonl_bytes, media_type="application/jsonl")
    report_pid, record_id, _record = store_descriptive_report(
        store,
        report_bytes,
        corpus_payload_id=corpus_pid,
        run=run,
    )

    report_path = dest / "report.json"
    findings_path = dest / "findings.md"
    manifest_path = dest / "run_manifest.json"
    report_path.write_bytes(report_bytes)
    findings_path.write_text(findings, encoding="utf-8")
    manifest = _run_manifest(
        corpus_id=spec.corpus_id,
        jsonl_path=source,
        corpus_payload_id=corpus_pid,
        report_payload_id=report_pid,
        record_id=record_id,
        run_id=run.run_id,
        code_ref=code_ref,
        figure_paths=figure_paths,
        started_at=started_at,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return AnalysisRunResult(
        report_path=report_path,
        findings_path=findings_path,
        manifest_path=manifest_path,
        figure_paths=figure_paths,
        corpus_payload_id=corpus_pid,
        report_payload_id=report_pid,
        record_id=record_id,
        run_id=run.run_id,
    )


def run_corpus_robustness(
    settings: Settings,
    *,
    corpus_id: str = "bgg_boardgames_v1",
    jsonl_path: Path | None = None,
    output_dir: Path | None = None,
    descriptive_report_payload_id: str | None = None,
) -> RobustnessRunResult:
    spec = corpus_spec(corpus_id)
    source = (
        jsonl_path
        if jsonl_path is not None
        else default_jsonl_path(settings.data_dir, corpus_id)
    )
    if not source.is_file():
        msg = (
            f"corpus JSONL not found: {source}. Ingest first with "
            f"`board-game-ingest --corpus --corpus-id {corpus_id}`."
        )
        raise FileNotFoundError(msg)
    started_at = datetime.now(UTC)
    loaded = load_games_jsonl(source)
    jsonl_bytes = source.read_bytes()
    corpus_pid = payload_id(jsonl_bytes)
    tables = build_robustness_tables(loaded.games)
    report = build_robustness_report(
        loaded.games,
        tables,
        corpus_id=spec.corpus_id,
        jsonl_name=source.name,
        corpus_payload_id=corpus_pid,
        started_at=started_at,
        load_errors=len(loaded.errors),
        ingest_counts=_ingest_counts(settings.data_dir, spec.corpus_id),
        descriptive_report_payload_id=descriptive_report_payload_id,
    )
    report_bytes = report_json_bytes(report)
    findings = render_robustness_findings(report)

    dest = (
        output_dir
        if output_dir is not None
        else default_robustness_output_dir(settings.data_dir)
    )
    dest.mkdir(parents=True, exist_ok=True)
    figures_dir = dest / "figures"
    figure_paths = write_robustness_figures(
        build_rows(loaded.games), tables, figures_dir
    )

    run = bga_run_context(started_at=started_at)
    code_ref = try_git_code_ref()
    if code_ref is not None:
        run = run.model_copy(update={"code_ref": CodeRef.model_validate(code_ref)})
    store = artifact_store(settings.data_dir)
    store.put(corpus_pid, jsonl_bytes, media_type="application/jsonl")
    report_pid, record_id, _record = store_descriptive_report(
        store,
        report_bytes,
        corpus_payload_id=corpus_pid,
        run=run,
        logical_key=ROBUSTNESS_LOGICAL_KEY,
        schema_path=default_robustness_schema_path(),
    )

    report_path = dest / "report.json"
    findings_path = dest / "robustness_findings.md"
    manifest_path = dest / "run_manifest.json"
    report_path.write_bytes(report_bytes)
    findings_path.write_text(findings, encoding="utf-8")
    manifest = _robustness_run_manifest(
        corpus_id=spec.corpus_id,
        jsonl_path=source,
        corpus_payload_id=corpus_pid,
        report_payload_id=report_pid,
        record_id=record_id,
        run_id=run.run_id,
        code_ref=code_ref,
        figure_paths=figure_paths,
        started_at=started_at,
        descriptive_report_payload_id=descriptive_report_payload_id,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return RobustnessRunResult(
        report_path=report_path,
        findings_path=findings_path,
        manifest_path=manifest_path,
        figure_paths=figure_paths,
        corpus_payload_id=corpus_pid,
        report_payload_id=report_pid,
        record_id=record_id,
        run_id=run.run_id,
    )


def _ingest_counts(data_dir: Path, corpus_id: str) -> dict[str, int] | None:
    path = data_dir / "derived" / "corpus" / f"{corpus_id}.manifest.json"
    if not path.is_file():
        return None
    payload = load_manifest(path)
    raw = payload.get("counts")
    if not isinstance(raw, dict):
        return None
    counts: dict[str, int] = {}
    for key in ("requested", "ok", "skipped", "errors"):
        value = raw.get(key)
        if isinstance(value, int):
            counts[key] = value
    return counts or None


def _run_manifest(
    *,
    corpus_id: str,
    jsonl_path: Path,
    corpus_payload_id: str,
    report_payload_id: str,
    record_id: str,
    run_id: str,
    code_ref: dict[str, Any] | None,
    figure_paths: list[Path],
    started_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": ANALYSIS_VERSION,
        "logical_key": ANALYSIS_LOGICAL_KEY,
        "corpus_id": corpus_id,
        "jsonl": str(jsonl_path),
        "corpus_payload_id": corpus_payload_id,
        "report_payload_id": report_payload_id,
        "record_id": record_id,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "figures": [str(path) for path in figure_paths],
    }
    if code_ref is not None:
        payload["code_ref"] = code_ref
    return payload


def _robustness_run_manifest(
    *,
    corpus_id: str,
    jsonl_path: Path,
    corpus_payload_id: str,
    report_payload_id: str,
    record_id: str,
    run_id: str,
    code_ref: dict[str, Any] | None,
    figure_paths: list[Path],
    started_at: datetime,
    descriptive_report_payload_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "analysis_id": ROBUSTNESS_ID,
        "analysis_version": ROBUSTNESS_VERSION,
        "logical_key": ROBUSTNESS_LOGICAL_KEY,
        "corpus_id": corpus_id,
        "jsonl": str(jsonl_path),
        "corpus_payload_id": corpus_payload_id,
        "report_payload_id": report_payload_id,
        "record_id": record_id,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "figures": [str(path) for path in figure_paths],
    }
    if descriptive_report_payload_id is not None:
        payload["descriptive_report_payload_id"] = descriptive_report_payload_id
    if code_ref is not None:
        payload["code_ref"] = code_ref
    return payload
