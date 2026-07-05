from __future__ import annotations

import os
from typing import Any


ENVIRONMENT_MODES = ["自動判定", "Cloud公開版", "ローカル高品質版"]


def should_reload_modules(debug_env: str | None = None) -> bool:
    """Reload project modules only in explicit development/debug mode."""
    value = os.environ.get("DEBUG_MODE", "") if debug_env is None else debug_env
    return str(value).strip().lower() in {"1", "true", "yes", "on", "debug"}


def resolve_environment_mode(environment_mode: str, cloud_detected: bool) -> dict[str, Any]:
    requested = environment_mode if environment_mode in ENVIRONMENT_MODES else "自動判定"
    cloud = bool(cloud_detected)
    warning = ""

    if requested == "自動判定":
        effective = "Cloud公開版" if cloud else "ローカル高品質版"
    elif requested == "Cloud公開版":
        effective = "Cloud公開版"
    else:
        if cloud:
            effective = "Cloud公開版"
            warning = "Cloud上ではローカル高品質版は利用できないため、Cloud公開版として動作します。"
        else:
            effective = "ローカル高品質版"

    public_prediction_only = effective == "Cloud公開版"
    return {
        "requested_mode": requested,
        "effective_mode": effective,
        "cloud_detected": cloud,
        "public_prediction_only": public_prediction_only,
        "allow_heavy_features": not public_prediction_only and not cloud,
        "force_public_checkbox": public_prediction_only,
        "warning": warning,
    }
