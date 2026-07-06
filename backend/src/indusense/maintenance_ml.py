"""Entraînement tabulaire pour la maintenance prédictive depuis le Gold Dataset."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

DEFAULT_LABEL = "label_failure_next_24h"
IDENTITY_COLUMNS = {"machine_id_std", "window_start", "window_end", "split_set"}
LEAKAGE_PREFIXES = ("label_failure_next_", "future_incident_count_")
DATETIME_HINTS = ("_at", "_date", "_time", "timestamp", "window_start", "window_end")


@dataclass(frozen=True)
class MaintenanceMlConfig:
    """Paramètres d'un run de maintenance prédictive."""

    gold_dir: Path
    run_dir: Path
    label_column: str = DEFAULT_LABEL
    gold_run_name: str | None = None
    random_forest_balanced: bool = True
    random_state: int = 42


def train_maintenance_models(config: MaintenanceMlConfig) -> dict[str, Any]:
    """Entraîne les modèles demandés et persiste un rapport JSON exploitable par l'API."""

    csv_path = resolve_gold_csv(config.gold_dir, config.gold_run_name)
    LOGGER.info("Chargement du Gold Dataset depuis %s.", csv_path)
    gold = pd.read_csv(csv_path)
    gold["window_start"] = pd.to_datetime(gold["window_start"], errors="coerce")
    gold = gold.sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)

    validate_gold_for_ml(gold, config.label_column)
    feature_columns = select_feature_columns(gold, config.label_column)
    split_frames = {
        split: gold.loc[gold["split_set"] == split].copy()
        for split in ("train", "validation", "test")
    }
    for split, frame in split_frames.items():
        if frame.empty:
            raise ValueError(f"Le split_set '{split}' est vide dans le Gold Dataset.")

    y_train = split_frames["train"][config.label_column].astype(int)
    y_validation = split_frames["validation"][config.label_column].astype(int)
    y_test = split_frames["test"][config.label_column].astype(int)
    x_train = split_frames["train"][feature_columns]
    x_validation = split_frames["validation"][feature_columns]
    x_test = split_frames["test"][feature_columns]

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = float(neg_count / pos_count) if pos_count else 1.0
    class_balance = {
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_validation.mean()),
        "test_positive_rate": float(y_test.mean()),
        "train_positive_count": pos_count,
        "train_negative_count": neg_count,
    }
    LOGGER.info("Taux de panne train: %.4f (%s positifs / %s lignes).", y_train.mean(), pos_count, len(y_train))

    preprocessor_linear = build_preprocessor(x_train, scale_numeric=True)
    preprocessor_tree = build_preprocessor(x_train, scale_numeric=False)
    models = build_models(config.random_state, scale_pos_weight, config.random_forest_balanced)
    mlflow_uri = setup_mlflow(config.run_dir)

    results = []
    models_dir = config.run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model, scale_numeric in models:
        LOGGER.info("Entraînement du modèle %s.", model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor_linear if scale_numeric else preprocessor_tree),
                ("model", model),
            ]
        )
        cv_metrics = temporal_cross_validate(pipeline, x_train, y_train)
        pipeline.fit(x_train, y_train)
        validation_scores = predict_scores(pipeline, x_validation)
        threshold = choose_threshold(y_validation, validation_scores)
        validation_metrics = evaluate_scores(y_validation, validation_scores, threshold)
        test_scores = predict_scores(pipeline, x_test)
        test_metrics = evaluate_scores(y_test, test_scores, threshold)
        model_path = models_dir / f"{model_name}.pkl"
        with model_path.open("wb") as model_file:
            pickle.dump(pipeline, model_file)

        log_mlflow_run(
            model_name=model_name,
            model=model,
            label_column=config.label_column,
            csv_path=csv_path,
            threshold=threshold,
            cv_metrics=cv_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            mlflow_uri=mlflow_uri,
        )
        results.append(
            {
                "model": model_name,
                "pr_auc_validation": validation_metrics["pr_auc"],
                "roc_auc_validation": validation_metrics["roc_auc"],
                "pr_auc_test": test_metrics["pr_auc"],
                "roc_auc_test": test_metrics["roc_auc"],
                "threshold": threshold,
                "cv_pr_auc_mean": cv_metrics["pr_auc_mean"],
                "cv_pr_auc_std": cv_metrics["pr_auc_std"],
                "cv_roc_auc_mean": cv_metrics["roc_auc_mean"],
                "cv_roc_auc_std": cv_metrics["roc_auc_std"],
                "validation_confusion_matrix": validation_metrics["confusion_matrix"],
                "test_confusion_matrix": test_metrics["confusion_matrix"],
                "model_path": str(model_path.relative_to(config.run_dir.parent.parent)),
            }
        )

    results = sorted(results, key=lambda row: row["pr_auc_validation"], reverse=True)
    best_model = results[0]["model"]
    report = {
        "run_id": config.run_dir.name,
        "status": "success",
        "created_at": utc_now(),
        "gold_run_name": csv_path.parent.name,
        "gold_csv_path": str(csv_path),
        "label_column": config.label_column,
        "rows": int(len(gold)),
        "features": len(feature_columns),
        "feature_columns": feature_columns,
        "class_balance": class_balance,
        "scale_pos_weight": scale_pos_weight,
        "random_forest_balanced": config.random_forest_balanced,
        "mlflow_tracking_uri": mlflow_uri,
        "results": results,
        "best_model": best_model,
        "conclusion": (
            f"{best_model} passe au module B7 car il obtient la meilleure PR-AUC validation "
            "sur un problème déséquilibré, avec un seuil choisi sur validation."
        ),
    }
    (config.run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def resolve_gold_csv(gold_dir: Path, run_name: str | None = None) -> Path:
    """Retourne le CSV Gold choisi, ou le dernier produit par l'ingestion."""

    if run_name:
        if "/" in run_name or "\\" in run_name or ".." in run_name:
            raise ValueError("Nom de run Gold invalide.")
        candidates = sorted((gold_dir / run_name).glob("gold_dataset_*.csv"), reverse=True)
    else:
        candidates = sorted(gold_dir.glob("*_gold_dataset/gold_dataset_*.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError("Aucun CSV Gold généré par l'ingestion n'a été trouvé.")
    return candidates[0]


def validate_gold_for_ml(gold: pd.DataFrame, label_column: str) -> None:
    required = {"machine_id_std", "window_start", "split_set", label_column}
    missing = sorted(required - set(gold.columns))
    if missing:
        raise ValueError(f"Colonnes Gold manquantes pour le ML: {', '.join(missing)}")
    expected_splits = {"train", "validation", "test"}
    missing_splits = sorted(expected_splits - set(gold["split_set"].dropna().astype(str)))
    if missing_splits:
        raise ValueError(f"split_set incomplet: {', '.join(missing_splits)}")
    if gold[label_column].isna().any():
        raise ValueError(f"La cible {label_column} contient des valeurs manquantes.")


def select_feature_columns(gold: pd.DataFrame, label_column: str) -> list[str]:
    """Exclut identifiants, labels et colonnes de fuite, puis garde les variables utilisables."""

    excluded = set(IDENTITY_COLUMNS)
    excluded.update(column for column in gold.columns if column.startswith(LEAKAGE_PREFIXES))
    excluded.add(label_column)
    features = []
    for column in gold.columns:
        if column in excluded:
            continue
        lower = column.lower()
        if any(hint in lower for hint in DATETIME_HINTS):
            continue
        if gold[column].nunique(dropna=True) <= 1:
            continue
        features.append(column)
    if not features:
        raise ValueError("Aucune feature exploitable après exclusion des colonnes de fuite.")
    return features


def build_preprocessor(frame: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(random_state: int, scale_pos_weight: float, random_forest_balanced: bool) -> list[tuple[str, Any, bool]]:
    models: list[tuple[str, Any, bool]] = [
        (
            "logistic_regression",
            LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"),
            True,
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=60,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced" if random_forest_balanced else None,
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        ),
    ]
    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=2,
                ),
                False,
            )
        )
    except Exception as error:  # noqa: BLE001 - dépendance optionnelle en environnement local.
        LOGGER.warning("XGBoost indisponible, le modèle xgboost est ignoré : %s", error)
    return models


def temporal_cross_validate(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    splits = min(3, max(2, int(y_train.sum())))
    splitter = TimeSeriesSplit(n_splits=splits)
    pr_auc_scores = []
    roc_auc_scores = []
    for train_index, validation_index in splitter.split(x_train):
        y_fold_train = y_train.iloc[train_index]
        y_fold_validation = y_train.iloc[validation_index]
        if y_fold_train.nunique() < 2 or y_fold_validation.nunique() < 2:
            continue
        pipeline.fit(x_train.iloc[train_index], y_fold_train)
        scores = predict_scores(pipeline, x_train.iloc[validation_index])
        pr_auc_scores.append(float(average_precision_score(y_fold_validation, scores)))
        roc_auc_scores.append(safe_roc_auc(y_fold_validation, scores))
    return {
        "pr_auc_mean": safe_mean(pr_auc_scores),
        "pr_auc_std": safe_std(pr_auc_scores),
        "roc_auc_mean": safe_mean(roc_auc_scores),
        "roc_auc_std": safe_std(roc_auc_scores),
        "folds_used": len(pr_auc_scores),
    }


def predict_scores(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(frame)[:, 1]
    decision = pipeline.decision_function(frame)
    return 1 / (1 + np.exp(-decision))


def choose_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def evaluate_scores(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": safe_roc_auc(y_true, scores),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }


def setup_mlflow(run_dir: Path) -> str:
    tracking_dir = run_dir.parent / "mlflow"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}"


def log_mlflow_run(
    model_name: str,
    model: Any,
    label_column: str,
    csv_path: Path,
    threshold: float,
    cv_metrics: dict[str, float],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    mlflow_uri: str,
) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("maintenance_predictive_b5")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("label_column", label_column)
            mlflow.log_param("gold_csv", str(csv_path))
            mlflow.log_param("threshold", threshold)
            for key, value in model.get_params().items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    mlflow.log_param(key, value)
            for key, value in cv_metrics.items():
                mlflow.log_metric(f"cv_{key}", float(value))
            mlflow.log_metric("validation_pr_auc", validation_metrics["pr_auc"])
            mlflow.log_metric("validation_roc_auc", validation_metrics["roc_auc"])
            mlflow.log_metric("test_pr_auc", test_metrics["pr_auc"])
            mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])
    except Exception as error:  # noqa: BLE001 - le run ML reste exploitable sans serveur MLflow.
        LOGGER.warning("Journalisation MLflow ignorée pour %s : %s", model_name, error)


def safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_mean(values: list[float]) -> float:
    return float(np.nanmean(values)) if values else float("nan")


def safe_std(values: list[float]) -> float:
    return float(np.nanstd(values)) if values else float("nan")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
