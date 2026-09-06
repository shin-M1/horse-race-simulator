from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.prediction_explainability import (  # noqa: E402,F401
    build_ai_evaluation_table,
    build_prediction_confidence,
    build_prediction_confidence_table,
    build_prediction_reason_table,
    build_shap_like_contributions,
    classify_racing_affiliation,
    compute_track_speed_index,
    contribution_table_from_row,
    enhance_prediction_table,
    field_strength_score,
    prepare_prediction_features,
    walk_forward_validation_splits,
)


__all__ = [
    "build_ai_evaluation_table",
    "build_prediction_confidence",
    "build_prediction_confidence_table",
    "build_prediction_reason_table",
    "build_shap_like_contributions",
    "classify_racing_affiliation",
    "compute_track_speed_index",
    "contribution_table_from_row",
    "enhance_prediction_table",
    "field_strength_score",
    "prepare_prediction_features",
    "walk_forward_validation_splits",
]
