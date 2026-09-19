"""Analysis specifications and the first corpus-descriptive workload."""

from board_game_analysis.analysis.descriptive import (
    ANALYSIS_ID,
    ANALYSIS_LOGICAL_KEY,
    ANALYSIS_VERSION,
    build_tables,
    player_profile,
    spearman_rho,
)
from board_game_analysis.analysis.run import run_corpus_descriptive
from board_game_analysis.analysis.spec import (
    EvidenceLevel,
    MeasurementRole,
    MeasurementScope,
    MeasurementSpec,
    MeasurementStatus,
    measurement_catalog,
    spec_by_id,
)

__all__ = [
    "ANALYSIS_ID",
    "ANALYSIS_LOGICAL_KEY",
    "ANALYSIS_VERSION",
    "EvidenceLevel",
    "MeasurementRole",
    "MeasurementScope",
    "MeasurementSpec",
    "MeasurementStatus",
    "build_tables",
    "measurement_catalog",
    "player_profile",
    "run_corpus_descriptive",
    "spearman_rho",
    "spec_by_id",
]
