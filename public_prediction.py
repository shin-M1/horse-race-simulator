from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.public_prediction import (  # noqa: E402
    PUBLIC_ENGINE_NAME,
    assign_evaluation_rank,
    build_public_prediction_result,
    calculate_ability_score,
    calculate_final_public_score,
    calculate_race_suitability_score,
    estimate_probabilities_from_scores,
    should_use_public_prediction,
)


__all__ = [
    "PUBLIC_ENGINE_NAME",
    "assign_evaluation_rank",
    "build_public_prediction_result",
    "calculate_ability_score",
    "calculate_final_public_score",
    "calculate_race_suitability_score",
    "estimate_probabilities_from_scores",
    "should_use_public_prediction",
]
