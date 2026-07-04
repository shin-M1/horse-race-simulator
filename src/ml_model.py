from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weight_optimizer import build_training_dataset


MODEL_DIR = Path("outputs/models")
TOP3_MODEL_PATH = MODEL_DIR / "top3_model.pkl"
WIN_MODEL_PATH = MODEL_DIR / "win_model.pkl"
ML_FEATURES = [
    "horse_ability_score",
    "race_strength_score",
    "elo_score",
    "late_kick_score",
    "course_fit_score",
    "pace_fit_score",
    "jockey_score",
    "track_bias_fit_score",
    "race_trend_score",
    "weight_penalty",
    "mud_aptitude",
    "finish_score",
    "margin_score",
    "time_score",
    "last3f_score",
    "carried_weight",
    "frame",
    "horse_number",
]


def build_ml_dataset(
    evaluation_logs: list[dict[str, Any]],
    target: str = "is_top3",
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in {"is_top3", "is_win"}:
        raise ValueError("target must be is_top3 or is_win")
    training = build_training_dataset(evaluation_logs)
    if training.empty:
        return pd.DataFrame(columns=ML_FEATURES), pd.Series(dtype=int, name=target)
    X = training.reindex(columns=ML_FEATURES).apply(pd.to_numeric, errors="coerce").fillna(50.0)
    y = pd.to_numeric(training[target], errors="coerce").fillna(0).astype(int)
    y.name = target
    return X, y


class FallbackLogisticModel:
    """Small numpy logistic model used when scikit-learn is unavailable."""

    def __init__(self) -> None:
        self.coef_: np.ndarray = np.zeros((1, 0), dtype=float)
        self.intercept_: np.ndarray = np.zeros(1, dtype=float)
        self.feature_names_in_: np.ndarray = np.array([], dtype=object)
        self.mean_: np.ndarray = np.array([], dtype=float)
        self.scale_: np.ndarray = np.array([], dtype=float)
        self.constant_probability_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FallbackLogisticModel":
        values = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        normalized = (values - self.mean_) / self.scale_
        if len(np.unique(targets)) < 2:
            self.constant_probability_ = float(targets.mean()) if len(targets) else 0.5
            self.coef_ = np.zeros((1, values.shape[1]), dtype=float)
            return self
        weights = np.zeros(values.shape[1], dtype=float)
        bias = 0.0
        learning_rate = 0.08
        for _ in range(600):
            logits = np.clip(normalized @ weights + bias, -20.0, 20.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            error = probabilities - targets
            weights -= learning_rate * ((normalized.T @ error) / len(targets) + 0.001 * weights)
            bias -= learning_rate * float(error.mean())
        self.coef_ = weights.reshape(1, -1)
        self.intercept_ = np.asarray([bias])
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if self.constant_probability_ is not None:
            positive = np.full(len(values), self.constant_probability_, dtype=float)
        else:
            normalized = (values - self.mean_) / self.scale_
            logits = np.clip(normalized @ self.coef_[0] + self.intercept_[0], -20.0, 20.0)
            positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_prediction_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "logistic",
) -> object:
    if X.empty or y.empty or len(X) != len(y):
        raise ValueError("training data is empty or inconsistent")
    model_type = str(model_type).lower()
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(n_estimators=150, random_state=42).fit(X, y)
        except ImportError:
            model_type = "logistic"
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(n_estimators=150, max_depth=4, random_state=42).fit(X, y)
        except ImportError:
            model_type = "logistic"
    try:
        if len(pd.Series(y).unique()) < 2:
            raise ValueError("single class")
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, random_state=42, class_weight="balanced")
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        return model.fit(X, y)
    except (ImportError, ValueError):
        return FallbackLogisticModel().fit(X, y)


def save_model(model: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(model, target)
    except ImportError:
        with target.open("wb") as handle:
            pickle.dump(model, handle)


def load_model(path: str | Path) -> object | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        import joblib

        return joblib.load(target)
    except ImportError:
        try:
            with target.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, pickle.PickleError):
            return None
    except Exception:
        return None


def resolve_prediction_engine(
    requested_engine: str,
    model_path: str | Path = TOP3_MODEL_PATH,
) -> str:
    requested = str(requested_engine or "rule_based")
    if requested == "ml_model" and not Path(model_path).is_file():
        return "rule_based"
    if requested not in {"rule_based", "optimized_weights", "ml_model"}:
        return "rule_based"
    return requested


def apply_ml_prediction(
    prediction_table: pd.DataFrame,
    top3_model: object,
    win_model: object | None = None,
) -> pd.DataFrame:
    if prediction_table.empty:
        return prediction_table.copy()
    table = prediction_table.copy()
    X = _features_from_prediction_table(table)
    top3_probability = _positive_probability(top3_model, X)
    win_probability = _positive_probability(win_model, X) if win_model is not None else top3_probability * 0.35
    table["ml_top3_probability"] = np.clip(top3_probability, 0.0, 1.0).round(5)
    table["ml_win_probability"] = np.clip(win_probability, 0.0, 1.0).round(5)
    table["prediction_score"] = np.clip((top3_probability * 0.70 + win_probability * 0.30) * 100.0, 0.0, 100.0).round(2)
    table["score"] = table["prediction_score"]
    table["prediction_engine"] = "ml_model"
    table = table.sort_values(
        ["prediction_score", "win_rate", "top3_rate", "avg_finish"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    marks = ("◎", "○", "▲", "△", "☆")
    table["印"] = [marks[index] if index < min(5, len(table)) else "" for index in range(len(table))]
    return table


def model_feature_importance(model: object, feature_names: list[str] | None = None) -> pd.DataFrame:
    names = feature_names or ML_FEATURES
    candidate = model
    if hasattr(model, "named_steps"):
        candidate = list(model.named_steps.values())[-1]
    if hasattr(candidate, "feature_importances_"):
        values = np.asarray(candidate.feature_importances_, dtype=float)
    elif hasattr(candidate, "coef_"):
        values = np.abs(np.asarray(candidate.coef_, dtype=float)[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    usable_names = names[: len(values)]
    return pd.DataFrame({"feature": usable_names, "importance": values[: len(usable_names)]}).sort_values(
        "importance", ascending=False
    )


def _features_from_prediction_table(table: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "elo_score": "normalized_elo_score",
        "carried_weight": "斤量",
        "frame": "枠順",
        "horse_number": "馬番",
    }
    features: dict[str, pd.Series] = {}
    for feature in ML_FEATURES:
        source = feature if feature in table.columns else aliases.get(feature, "")
        if source and source in table.columns:
            features[feature] = pd.to_numeric(table[source], errors="coerce").fillna(50.0)
        else:
            features[feature] = pd.Series(50.0, index=table.index)
    return pd.DataFrame(features, index=table.index)


def _positive_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        return probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities.reshape(-1)
    return np.asarray(model.predict(X), dtype=float)
