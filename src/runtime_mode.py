from __future__ import annotations

import os


def should_reload_modules(debug_env: str | None = None) -> bool:
    """Reload project modules only in explicit development/debug mode."""
    value = os.environ.get("DEBUG_MODE", "") if debug_env is None else debug_env
    return str(value).strip().lower() in {"1", "true", "yes", "on", "debug"}


def is_cloud_environment() -> bool:
    return os.environ.get("STREAMLIT_RUNTIME_ENV") is not None or os.environ.get("HOSTNAME", "").startswith("streamlit")


def get_runtime_mode(cloud_detected: bool | None = None) -> str:
    cloud = is_cloud_environment() if cloud_detected is None else bool(cloud_detected)
    return "CLOUD" if cloud else "LOCAL"
