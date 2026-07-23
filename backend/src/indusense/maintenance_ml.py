"""Entraînement tabulaire pour la maintenance prédictive depuis le Gold Dataset."""

from __future__ import annotations

import json
import logging
import hashlib
import platform
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from indusense.maintenance.carbon import MaintenanceEmissionsTracker
from indusense.maintenance.tune import run_xgboost_study

LOGGER = logging.getLogger(__name__)

DEFAULT_LABEL = "label_failure_next_24h"
DEFAULT_SELECTED_MODELS = ("logistic_regression", "decision_tree", "random_forest", "random_forest_balanced", "xgboost")
SUPPORTED_MODELS = set(DEFAULT_SELECTED_MODELS)
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
    selected_models: tuple[str, ...] = DEFAULT_SELECTED_MODELS
    decision_tree_max_depth: int = 6
    decision_tree_min_samples_leaf: int = 10
    random_forest_n_estimators: int = 60
    random_forest_max_depth: int = 12
    random_forest_min_samples_leaf: int = 2
    random_forest_min_samples_split: int = 10
    random_forest_max_features: str | None = "sqrt"
    random_forest_bootstrap: bool = True
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_scale_pos_weight_auto: bool = True
    xgboost_scale_pos_weight: float | None = None
    threshold_strategy: str = "balanced"
    target_recall: float = 0.8
    false_negative_cost: float = 20.0
    false_positive_cost: float = 1.0
    experiment_hypothesis: str = ""
    random_state: int = 42
    tune: bool = False
    tune_n_trials: int = 15
    tune_timeout_seconds: int = 600
    tune_mode: str = "frugal"


def train_maintenance_models(config: MaintenanceMlConfig) -> dict[str, Any]:
    """Entraîne les modèles demandés et persiste un rapport JSON exploitable par l'API."""

    event_log = TrainingEventLog(config.run_dir / "training_events.jsonl")
    event_log.write("start", "running", "Démarrage du run ML.", label_column=config.label_column, models=list(config.selected_models))
    csv_path = resolve_gold_csv(config.gold_dir, config.gold_run_name)
    LOGGER.info("Chargement du Gold Dataset depuis %s.", csv_path)
    gold = pd.read_csv(csv_path)
    gold["window_start"] = pd.to_datetime(gold["window_start"], errors="coerce")
    gold = gold.sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)
    dataset_hash = hash_dataframe(gold)
    event_log.write("load_dataset", "success", "Gold dataset chargé.", rows=len(gold), columns=gold.shape[1], dataset_hash=dataset_hash, source=str(csv_path))

    validate_gold_for_ml(gold, config.label_column)
    feature_columns = select_feature_columns(gold, config.label_column)
    event_log.write("prepare_features", "success", "Variables explicatives préparées.", features=len(feature_columns), excluded_labels=config.label_column)
    LOGGER.info(
        "Réglages ML: modèles=%s, seuil=%s, arbre(depth=%s, leaf=%s), random_forest(n=%s, depth=%s, leaf=%s, split=%s, max_features=%s, bootstrap=%s, balanced=%s).",
        ", ".join(config.selected_models),
        config.threshold_strategy,
        config.decision_tree_max_depth,
        config.decision_tree_min_samples_leaf,
        config.random_forest_n_estimators,
        config.random_forest_max_depth,
        config.random_forest_min_samples_leaf,
        config.random_forest_min_samples_split,
        config.random_forest_max_features,
        config.random_forest_bootstrap,
        config.random_forest_balanced,
    )
    split_frames = {
        split: gold.loc[gold["split_set"] == split].copy()
        for split in ("train", "validation", "test")
    }
    for split, frame in split_frames.items():
        if frame.empty:
            raise ValueError(f"Le split_set '{split}' est vide dans le Gold Dataset.")
    event_log.write(
        "split",
        "success",
        "Découpage temporel train / validation / test validé.",
        train_rows=len(split_frames["train"]),
        validation_rows=len(split_frames["validation"]),
        test_rows=len(split_frames["test"]),
    )

    y_train = split_frames["train"][config.label_column].astype(int)
    y_validation = split_frames["validation"][config.label_column].astype(int)
    y_test = split_frames["test"][config.label_column].astype(int)
    x_train = split_frames["train"][feature_columns]
    x_validation = split_frames["validation"][feature_columns]
    x_test = split_frames["test"][feature_columns]

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = float(neg_count / pos_count) if pos_count else 1.0
    xgboost_effective_scale_pos_weight = scale_pos_weight if config.xgboost_scale_pos_weight_auto else float(config.xgboost_scale_pos_weight or 1.0)
    class_balance = {
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_validation.mean()),
        "test_positive_rate": float(y_test.mean()),
        "train_positive_count": pos_count,
        "train_negative_count": neg_count,
    }
    LOGGER.info("Taux de panne train: %.4f (%s positifs / %s lignes).", y_train.mean(), pos_count, len(y_train))
    event_log.write(
        "class_balance",
        "success",
        "Déséquilibre des classes mesuré.",
        train_positive_rate=class_balance["train_positive_rate"],
        train_positive_count=pos_count,
        train_negative_count=neg_count,
    )

    preprocessor_linear = build_preprocessor(x_train, scale_numeric=True)
    preprocessor_tree = build_preprocessor(x_train, scale_numeric=False)
    models = build_models(config, xgboost_effective_scale_pos_weight)
    if not models:
        raise ValueError("Aucun modèle sélectionné pour l'entraînement.")
    mlflow_uri = setup_mlflow(config.run_dir)
    artifacts_dir = config.run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    tuning: dict[str, Any] = {"enabled": False}
    if config.tune:
        xgb_entry = next((entry for entry in models if entry[0] == "xgboost"), None)
        if xgb_entry is None:
            tuning = {"enabled": True, "available": False, "reason": "XGBoost doit être sélectionné pour lancer Optuna."}
        else:
            _, xgb_model, xgb_scale = xgb_entry
            xgb_pipeline = Pipeline(steps=[("preprocess", preprocessor_linear if xgb_scale else preprocessor_tree), ("model", xgb_model)])
            tuning = run_xgboost_study(
                xgb_pipeline, x_train, y_train, n_trials=config.tune_n_trials,
                timeout_seconds=config.tune_timeout_seconds, seed=config.random_state,
                aggressive_pruning=config.tune_mode == "frugal", output_dir=artifacts_dir / "optuna",
            )
            tuning["enabled"] = True
            tuning["mode"] = config.tune_mode
            if tuning.get("best_params"):
                xgb_model.set_params(**{key.removeprefix("model__"): value for key, value in tuning["best_params"].items()})
            for key, value in (tuning.get("artifacts") or {}).items():
                if value:
                    path = Path(value)
                    if path.exists():
                        tuning["artifacts"][key] = str(path.relative_to(config.run_dir))

    results = []
    models_dir = config.run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    carbon_tracker = MaintenanceEmissionsTracker(config.run_dir)
    carbon_tracker.__enter__()
    for model_name, model, scale_numeric in models:
        LOGGER.info("Entraînement du modèle %s.", model_name)
        event_log.write("train_model", "running", f"Entraînement du modèle {model_name}.", model=model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor_linear if scale_numeric else preprocessor_tree),
                ("model", model),
            ]
        )
        train_started_at = time.perf_counter()
        cv_metrics = temporal_cross_validate(pipeline, x_train, y_train)
        pipeline.fit(x_train, y_train)
        training_seconds = time.perf_counter() - train_started_at
        validation_scores = predict_scores(pipeline, x_validation)
        threshold = choose_threshold(y_validation, validation_scores, config.threshold_strategy, config.target_recall)
        event_log.write("threshold", "success", "Seuil calculé sur validation.", model=model_name, threshold=threshold, strategy=config.threshold_strategy)
        train_scores = predict_scores(pipeline, x_train)
        train_metrics = evaluate_scores(y_train, train_scores, threshold, config.false_negative_cost, config.false_positive_cost)
        validation_metrics = evaluate_scores(y_validation, validation_scores, threshold, config.false_negative_cost, config.false_positive_cost)
        inference_started_at = time.perf_counter()
        test_scores = predict_scores(pipeline, x_test)
        inference_seconds = time.perf_counter() - inference_started_at
        inference_ms_per_row = (inference_seconds / max(len(x_test), 1)) * 1000
        test_metrics = evaluate_scores(y_test, test_scores, threshold, config.false_negative_cost, config.false_positive_cost)
        test_metrics_at_05 = evaluate_scores(y_test, test_scores, 0.5, config.false_negative_cost, config.false_positive_cost)
        feature_insights = extract_feature_insights(pipeline)
        shap_explanations = build_shap_explanations(
            pipeline=pipeline,
            x_test=x_test,
            test_scores=test_scores,
            threshold=threshold,
            model_name=model_name,
        )
        model_artifacts = write_model_artifacts(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            y_validation=y_validation,
            validation_scores=validation_scores,
            validation_metrics=validation_metrics,
            y_test=y_test,
            test_scores=test_scores,
            test_metrics=test_metrics,
            threshold=threshold,
            feature_insights=feature_insights,
            shap_explanations=shap_explanations,
        )
        model_path = models_dir / f"{model_name}.pkl"
        with model_path.open("wb") as model_file:
            pickle.dump(pipeline, model_file)
        model_size_bytes = model_path.stat().st_size
        model_artifacts["model_pickle"] = str(model_path.relative_to(config.run_dir))

        mlflow_run_id = log_mlflow_run(
            model_name=model_name,
            pipeline=pipeline,
            label_column=config.label_column,
            csv_path=csv_path,
            dataset_hash=dataset_hash,
            threshold=threshold,
            cv_metrics=cv_metrics,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            artifacts=model_artifacts,
            run_dir=config.run_dir,
            mlflow_uri=mlflow_uri,
            threshold_strategy=config.threshold_strategy,
            target_recall=config.target_recall,
            false_negative_cost=config.false_negative_cost,
            false_positive_cost=config.false_positive_cost,
            experiment_hypothesis=config.experiment_hypothesis,
            random_state=config.random_state,
        )
        event_log.write(
            "evaluate_model",
            "success",
            f"Évaluation terminée pour {model_name}.",
            model=model_name,
            validation_pr_auc=validation_metrics["pr_auc"],
            test_pr_auc=test_metrics["pr_auc"],
            test_recall=test_metrics["recall"],
            business_cost=test_metrics["business_cost"],
        )
        results.append(
            {
                "model": model_name,
                "pr_auc_train": train_metrics["pr_auc"],
                "roc_auc_train": train_metrics["roc_auc"],
                "pr_auc_validation": validation_metrics["pr_auc"],
                "roc_auc_validation": validation_metrics["roc_auc"],
                "pr_auc_test": test_metrics["pr_auc"],
                "roc_auc_test": test_metrics["roc_auc"],
                "precision_validation": validation_metrics["precision"],
                "recall_validation": validation_metrics["recall"],
                "f1_validation": validation_metrics["f1"],
                "precision_test": test_metrics["precision"],
                "recall_test": test_metrics["recall"],
                "f1_test": test_metrics["f1"],
                "threshold": threshold,
                "business_cost_test": test_metrics["business_cost"],
                "training_seconds": float(training_seconds),
                "inference_ms_per_row": float(inference_ms_per_row),
                "model_size_bytes": int(model_size_bytes),
                "test_metrics_at_05": test_metrics_at_05,
                "cv_pr_auc_mean": cv_metrics["pr_auc_mean"],
                "cv_pr_auc_std": cv_metrics["pr_auc_std"],
                "cv_roc_auc_mean": cv_metrics["roc_auc_mean"],
                "cv_roc_auc_std": cv_metrics["roc_auc_std"],
                "validation_confusion_matrix": validation_metrics["confusion_matrix"],
                "test_confusion_matrix": test_metrics["confusion_matrix"],
                "top_features": feature_insights,
                "shap_explanations": shap_explanations,
                "artifacts": model_artifacts,
                "mlflow_run_id": mlflow_run_id,
                "model_path": str(model_path.relative_to(config.run_dir.parent.parent)),
            }
        )

    carbon_tracker.__exit__(None, None, None)
    results = sorted(results, key=lambda row: row["pr_auc_validation"], reverse=True)
    best_model = results[0]["model"]
    best_result = results[0]
    b7_artifacts = write_b7_arbitration_artifacts(artifacts_dir, results, carbon_tracker.result, tuning)
    reproducibility = {
        "dataset_hash": dataset_hash,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
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
        "pr_auc_random_baseline": class_balance["validation_positive_rate"],
        "scale_pos_weight": scale_pos_weight,
        "xgboost_effective_scale_pos_weight": xgboost_effective_scale_pos_weight,
        "random_forest_balanced": config.random_forest_balanced,
        "selected_models": list(config.selected_models),
        "decision_tree_max_depth": config.decision_tree_max_depth,
        "decision_tree_min_samples_leaf": config.decision_tree_min_samples_leaf,
        "random_forest_n_estimators": config.random_forest_n_estimators,
        "random_forest_max_depth": config.random_forest_max_depth,
        "random_forest_min_samples_leaf": config.random_forest_min_samples_leaf,
        "random_forest_min_samples_split": config.random_forest_min_samples_split,
        "random_forest_max_features": config.random_forest_max_features,
        "random_forest_bootstrap": config.random_forest_bootstrap,
        "xgboost_n_estimators": config.xgboost_n_estimators,
        "xgboost_max_depth": config.xgboost_max_depth,
        "xgboost_learning_rate": config.xgboost_learning_rate,
        "xgboost_scale_pos_weight_auto": config.xgboost_scale_pos_weight_auto,
        "xgboost_scale_pos_weight": config.xgboost_scale_pos_weight,
        "threshold_strategy": config.threshold_strategy,
        "target_recall": config.target_recall,
        "false_negative_cost": config.false_negative_cost,
        "false_positive_cost": config.false_positive_cost,
        "experiment_hypothesis": config.experiment_hypothesis,
        "random_state": config.random_state,
        "mlflow_tracking_uri": mlflow_uri,
        "tuning": tuning,
        "carbon": carbon_tracker.result,
        "b7_artifacts": b7_artifacts,
        "reproducibility": reproducibility,
        "event_log_path": str((config.run_dir / "training_events.jsonl").relative_to(config.run_dir.parent.parent)),
        "results": results,
        "best_model": best_model,
        "conclusion": (
            f"{best_model} passe au module B7 car il obtient la meilleure PR-AUC validation "
            "sur un problème déséquilibré, avec un seuil choisi sur validation."
        ),
    }
    event_log.write(
        "select_best_model",
        "success",
        "Meilleur modèle sélectionné sur PR-AUC validation.",
        model=best_model,
        threshold=best_result["threshold"],
        validation_pr_auc=best_result["pr_auc_validation"],
    )
    (config.run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    event_log.write("finish", "success", "Rapport ML généré.", report_path="report.json")
    return report


def write_b7_arbitration_artifacts(
    artifacts_dir: Path, results: list[dict[str, Any]], carbon: dict[str, Any], tuning: dict[str, Any],
) -> dict[str, str]:
    """Produit le tableau d'arbitrage et une lecture performance/CO₂ explicite."""
    rows = []
    share = (carbon.get("emissions_gco2eq") or 0.0) / max(len(results), 1)
    for item in results:
        rows.append({
            "model": item["model"], "pr_auc_validation": item["pr_auc_validation"],
            "pr_auc_test": item["pr_auc_test"], "gco2eq_estimated_share": share,
            "interpretability": "SHAP arbre disponible" if item.get("shap_explanations", {}).get("available") else "importance native / non disponible",
            "decision": "candidat" if item == results[0] else "comparateur",
        })
    path = artifacts_dir / "b7_arbitration.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    graph = artifacts_dir / "performance_vs_carbon.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(6, 4))
        x_values = [row["gco2eq_estimated_share"] for row in rows]
        y_values = [row["pr_auc_validation"] for row in rows]
        axis.scatter(x_values, y_values, color="#2563eb")
        for row, x_value, y_value in zip(rows, x_values, y_values):
            axis.annotate(row["model"], (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set(xlabel="gCO₂eq estimés (part du run)", ylabel="PR-AUC validation", title="Arbitrage performance / CO₂")
        axis.grid(alpha=0.25); figure.tight_layout(); figure.savefig(graph, dpi=150); plt.close(figure)
    except Exception:
        pass
    outcome = {"arbitration_csv": str(path.relative_to(artifacts_dir.parent))}
    if graph.exists(): outcome["performance_vs_carbon_png"] = str(graph.relative_to(artifacts_dir.parent))
    if tuning.get("enabled"): outcome["study_mode"] = str(tuning.get("mode", "frugal"))
    return outcome


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


def build_models(config: MaintenanceMlConfig, scale_pos_weight: float) -> list[tuple[str, Any, bool]]:
    selected_models = [model for model in config.selected_models if model in SUPPORTED_MODELS]
    models: list[tuple[str, Any, bool]] = []
    if "logistic_regression" in selected_models:
        models.append(
            (
                "logistic_regression",
                LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"),
                True,
            )
        )
    if "decision_tree" in selected_models:
        models.append(
            (
                "decision_tree",
                DecisionTreeClassifier(
                    max_depth=config.decision_tree_max_depth,
                    min_samples_leaf=config.decision_tree_min_samples_leaf,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
                False,
            )
        )
    if "random_forest" in selected_models:
        models.append(
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=config.random_forest_n_estimators,
                    max_depth=config.random_forest_max_depth,
                    min_samples_leaf=config.random_forest_min_samples_leaf,
                    min_samples_split=config.random_forest_min_samples_split,
                    max_features=normalize_random_forest_max_features(config.random_forest_max_features),
                    bootstrap=config.random_forest_bootstrap,
                    class_weight=None,
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
                False,
            )
        )
    if "random_forest_balanced" in selected_models:
        models.append(
            (
                "random_forest_balanced",
                RandomForestClassifier(
                    n_estimators=config.random_forest_n_estimators,
                    max_depth=config.random_forest_max_depth,
                    min_samples_leaf=config.random_forest_min_samples_leaf,
                    min_samples_split=config.random_forest_min_samples_split,
                    max_features=normalize_random_forest_max_features(config.random_forest_max_features),
                    bootstrap=config.random_forest_bootstrap,
                    class_weight="balanced",
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
                False,
            )
        )
    if "xgboost" in selected_models:
        try:
            from xgboost import XGBClassifier

            models.append(
                (
                    "xgboost",
                    XGBClassifier(
                        n_estimators=config.xgboost_n_estimators,
                        max_depth=config.xgboost_max_depth,
                        learning_rate=config.xgboost_learning_rate,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        scale_pos_weight=scale_pos_weight,
                        random_state=config.random_state,
                        n_jobs=2,
                    ),
                    False,
                )
            )
        except Exception as error:  # noqa: BLE001 - dépendance optionnelle en environnement local.
            LOGGER.warning("XGBoost indisponible, le modèle xgboost est ignoré : %s", error)
    return models


def normalize_random_forest_max_features(value: str | None) -> str | None:
    if value in {None, "", "all", "none"}:
        return None
    if value in {"sqrt", "log2"}:
        return value
    return "sqrt"


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


def choose_threshold(y_true: pd.Series, scores: np.ndarray, strategy: str, target_recall: float = 0.8) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    if strategy == "target_recall":
        candidates = np.where(recall[:-1] >= target_recall)[0]
        if len(candidates):
            best_precision = np.nanmax(precision[candidates])
            best_candidates = candidates[np.where(precision[candidates] == best_precision)[0]]
            best_index = int(best_candidates[-1])
            return float(thresholds[best_index])
    beta = {"recall": 2.0, "precision": 0.5}.get(strategy, 1.0)
    beta_squared = beta**2
    f_scores = (
        (1 + beta_squared)
        * precision[:-1]
        * recall[:-1]
        / np.maximum((beta_squared * precision[:-1]) + recall[:-1], 1e-12)
    )
    best_index = int(np.nanargmax(f_scores))
    return float(thresholds[best_index])


def evaluate_scores(y_true: pd.Series, scores: np.ndarray, threshold: float, false_negative_cost: float = 20.0, false_positive_cost: float = 1.0) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": safe_roc_auc(y_true, scores),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "business_cost": float(false_negatives * false_negative_cost + false_positives * false_positive_cost),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": false_positives,
            "fn": false_negatives,
            "tp": int(matrix[1, 1]),
        },
    }


class TrainingEventLog:
    """Journal JSONL lisible par l'interface pendant un run ML."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def write(self, step: str, status: str, message: str, **details: Any) -> None:
        event = {
            "ts": utc_now(),
            "step": step,
            "status": status,
            "message": message,
            "details": json_safe(details),
        }
        with self.path.open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(event, ensure_ascii=False) + "\n")


def hash_dataframe(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update("|".join(frame.columns).encode("utf-8"))
    return digest.hexdigest()[:12]


def write_model_artifacts(
    artifacts_dir: Path,
    model_name: str,
    y_validation: pd.Series,
    validation_scores: np.ndarray,
    validation_metrics: dict[str, Any],
    y_test: pd.Series,
    test_scores: np.ndarray,
    test_metrics: dict[str, Any],
    threshold: float,
    feature_insights: list[dict[str, Any]],
    shap_explanations: dict[str, Any],
) -> dict[str, str]:
    model_dir = artifacts_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "validation_confusion_matrix_csv": write_confusion_matrix_csv(model_dir / "validation_confusion_matrix.csv", validation_metrics["confusion_matrix"]),
        "test_confusion_matrix_csv": write_confusion_matrix_csv(model_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"]),
        "test_confusion_matrix_png": write_confusion_matrix_png(model_dir / "test_confusion_matrix.png", test_metrics["confusion_matrix"], f"{model_name} - test"),
        "precision_recall_curve_png": write_precision_recall_curve(model_dir / "precision_recall_curve.png", y_validation, validation_scores, threshold),
        "roc_curve_png": write_roc_curve(model_dir / "roc_curve.png", y_test, test_scores),
    }
    if shap_explanations.get("available"):
        shap_explanations["artifacts"] = write_shap_artifacts(model_dir, shap_explanations)
    shap_artifacts = shap_explanations.get("artifacts") or {}
    for key, path in shap_artifacts.items():
        if Path(path).exists():
            artifacts[key] = str(Path(path).relative_to(artifacts_dir.parent))
    if feature_insights:
        feature_path = model_dir / "feature_importances.csv"
        pd.DataFrame(feature_insights).to_csv(feature_path, index=False)
        artifacts["feature_importances_csv"] = str(feature_path.relative_to(artifacts_dir.parent))
    return artifacts


def write_confusion_matrix_csv(path: Path, matrix: dict[str, int]) -> str:
    pd.DataFrame(
        [
            {"actual": "normal", "predicted_normal": matrix["tn"], "predicted_failure": matrix["fp"]},
            {"actual": "failure", "predicted_normal": matrix["fn"], "predicted_failure": matrix["tp"]},
        ]
    ).to_csv(path, index=False)
    return str(path.relative_to(path.parents[2]))


def write_confusion_matrix_png(path: Path, matrix: dict[str, int], title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1], labels=["Pas d'alerte", "Alerte"])
    ax.set_yticks([0, 1], labels=["Normal", "Panne"])
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(values[row, col]), ha="center", va="center", color="#0b0d12", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(path.parents[2]))


def write_precision_recall_curve(path: Path, y_true: pd.Series, scores: np.ndarray, threshold: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    threshold_index = int(np.argmin(np.abs(thresholds - threshold))) if len(thresholds) else 0
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(recall, precision, label="Courbe PR")
    prevalence = float(y_true.mean())
    ax.axhline(prevalence, linestyle="--", color="#94a3b8", label=f"Aléatoire = prévalence {prevalence:.3f}")
    if len(thresholds):
        ax.scatter(recall[threshold_index], precision[threshold_index], color="#ef4444", label=f"Seuil {threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Précision")
    ax.set_title("Precision-Recall validation")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(path.parents[2]))


def build_shap_explanations(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    test_scores: np.ndarray,
    threshold: float,
    model_name: str,
    max_rows: int = 300,
) -> dict[str, Any]:
    """Produit les explications SHAP globales et locales quand la librairie est disponible."""

    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return {"available": False, "reason": "SHAP TreeExplainer est limité ici aux modèles arbre/boosting."}

    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001 - SHAP reste optionnel pour garder le run robuste.
        return {"available": False, "reason": f"Librairie SHAP indisponible : {error}"}

    try:
        preprocessor = pipeline.named_steps["preprocess"]
        feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
        sample = x_test.head(min(max_rows, len(x_test)))
        transformed = preprocessor.transform(sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)

        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(transformed)
        shap_values = normalize_binary_shap_values(raw_values)
        base_value = normalize_binary_base_value(explainer.expected_value)
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_features = [
            {"feature": str(feature_names[index]), "mean_abs_shap": float(mean_abs[index])}
            for index in order[:10]
            if mean_abs[index] > 0
        ]

        return {
            "available": True,
            "base_value": float(base_value),
            "top_features": top_features,
            "sample": build_local_shap_sample(sample, transformed, shap_values, test_scores, threshold, feature_names),
            "_raw": {
                "feature_names": feature_names.tolist(),
                "transformed": transformed,
                "shap_values": shap_values,
                "sample_index": int(select_local_sample_index(test_scores[: len(sample)], threshold)),
            },
        }
    except Exception as error:  # noqa: BLE001 - l'explication ne doit pas invalider l'entraînement.
        LOGGER.warning("SHAP indisponible pour %s : %s", model_name, error)
        return {"available": False, "reason": str(error)}


def write_shap_artifacts(model_dir: Path, shap_explanations: dict[str, Any]) -> dict[str, str]:
    raw = shap_explanations.pop("_raw", None)
    if not raw:
        return {}
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feature_names = raw["feature_names"]
        transformed = np.asarray(raw["transformed"], dtype=float)
        shap_values = np.asarray(raw["shap_values"], dtype=float)
        sample_index = int(raw["sample_index"])
        artifacts = {}

        summary_path = model_dir / "shap_summary.png"
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        artifacts["shap_summary_png"] = str(summary_path)

        local_path = model_dir / "shap_waterfall_sample.png"
        explanation = shap.Explanation(
            values=shap_values[sample_index],
            base_values=float(shap_explanations.get("base_value", 0.0)),
            data=transformed[sample_index],
            feature_names=feature_names,
        )
        shap.waterfall_plot(explanation, show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(local_path, dpi=150, bbox_inches="tight")
        plt.close()
        artifacts["shap_waterfall_png"] = str(local_path)
        return artifacts
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Écriture des artefacts SHAP impossible : %s", error)
        shap_explanations["available"] = False
        shap_explanations["reason"] = str(error)
        return {}


def normalize_binary_shap_values(raw_values: Any) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0], dtype=float)
    values = np.asarray(raw_values, dtype=float)
    if values.ndim == 3:
        return values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


def normalize_binary_base_value(raw_value: Any) -> float:
    values = np.asarray(raw_value, dtype=float)
    if values.ndim == 0:
        return float(values)
    return float(values[1] if values.size > 1 else values[0])


def select_local_sample_index(scores: np.ndarray, threshold: float) -> int:
    alert_indices = np.where(scores >= threshold)[0]
    if len(alert_indices):
        return int(alert_indices[np.argmax(scores[alert_indices])])
    return int(np.argmax(scores)) if len(scores) else 0


def build_local_shap_sample(
    sample: pd.DataFrame,
    transformed: np.ndarray,
    shap_values: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    feature_names: np.ndarray,
) -> dict[str, Any]:
    sample_index = select_local_sample_index(scores[: len(sample)], threshold)
    local_values = shap_values[sample_index]
    order = np.argsort(np.abs(local_values))[::-1][:5]
    factors = []
    for index in order:
        contribution = float(local_values[index])
        if contribution == 0:
            continue
        factors.append(
            {
                "feature": str(feature_names[index]),
                "value": float(transformed[sample_index, index]),
                "contribution": contribution,
                "direction": "augmente" if contribution > 0 else "réduit",
            }
        )
    return {
        "row_number": int(sample.index[sample_index]) if len(sample.index) else sample_index,
        "score": float(scores[sample_index]) if len(scores) else float("nan"),
        "threshold": float(threshold),
        "is_alert": bool(len(scores) and scores[sample_index] >= threshold),
        "factors": factors,
    }


def write_roc_curve(path: Path, y_true: pd.Series, scores: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    if y_true.nunique() >= 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Aléatoire")
    ax.set_xlabel("Faux positifs")
    ax.set_ylabel("Vrais positifs")
    ax.set_title("ROC test")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(path.parents[2]))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def setup_mlflow(run_dir: Path) -> str:
    tracking_dir = run_dir.parent / "mlflow"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}"


def log_mlflow_run(
    model_name: str,
    pipeline: Pipeline,
    label_column: str,
    csv_path: Path,
    dataset_hash: str,
    threshold: float,
    cv_metrics: dict[str, float],
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    artifacts: dict[str, str],
    run_dir: Path,
    mlflow_uri: str,
    threshold_strategy: str,
    target_recall: float,
    false_negative_cost: float,
    false_positive_cost: float,
    experiment_hypothesis: str,
    random_state: int,
) -> str | None:
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("maintenance_predictive_b5")
        with mlflow.start_run(run_name=model_name) as active_run:
            model = pipeline.named_steps["model"]
            mlflow.log_param("model", model_name)
            mlflow.log_param("label_column", label_column)
            mlflow.log_param("gold_csv", str(csv_path))
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("threshold_strategy", threshold_strategy)
            mlflow.log_param("target_recall", target_recall)
            mlflow.log_param("false_negative_cost", false_negative_cost)
            mlflow.log_param("false_positive_cost", false_positive_cost)
            mlflow.log_param("experiment_hypothesis", experiment_hypothesis)
            mlflow.log_param("random_state", random_state)
            mlflow.set_tag("dataset_hash", dataset_hash)
            mlflow.set_tag("app_run_id", run_dir.name)
            mlflow.set_tag("python_version", platform.python_version())
            mlflow.set_tag("sklearn_version", sklearn.__version__)
            mlflow.set_tag("tracking_profile", "manual+artifacts")
            for key, value in model.get_params().items():
                param_key = f"model_{key}"
                if isinstance(value, (str, int, float, bool, type(None))):
                    mlflow.log_param(param_key, value)
            for key, value in cv_metrics.items():
                mlflow.log_metric(f"cv_{key}", float(value))
            mlflow.log_metric("train_pr_auc", train_metrics["pr_auc"])
            mlflow.log_metric("train_roc_auc", train_metrics["roc_auc"])
            mlflow.log_metric("validation_pr_auc", validation_metrics["pr_auc"])
            mlflow.log_metric("validation_roc_auc", validation_metrics["roc_auc"])
            mlflow.log_metric("validation_precision", validation_metrics["precision"])
            mlflow.log_metric("validation_recall", validation_metrics["recall"])
            mlflow.log_metric("validation_f1", validation_metrics["f1"])
            mlflow.log_metric("test_pr_auc", test_metrics["pr_auc"])
            mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])
            mlflow.log_metric("test_precision", test_metrics["precision"])
            mlflow.log_metric("test_recall", test_metrics["recall"])
            mlflow.log_metric("test_f1", test_metrics["f1"])
            mlflow.log_metric("test_business_cost", test_metrics["business_cost"])
            for artifact in artifacts.values():
                artifact_path = Path(artifact)
                absolute_artifact_path = run_dir / artifact_path
                if absolute_artifact_path.exists():
                    mlflow.log_artifact(str(absolute_artifact_path), artifact_path.parent.as_posix())
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
            )
            return active_run.info.run_id
    except Exception as error:  # noqa: BLE001 - le run ML reste exploitable sans serveur MLflow.
        LOGGER.warning("Journalisation MLflow ignorée pour %s : %s", model_name, error)
        return None


def extract_feature_insights(pipeline: Pipeline, top_n: int = 10) -> list[dict[str, Any]]:
    """Retourne les variables les plus utiles pour expliquer simplement un modèle entraîné."""

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:  # noqa: BLE001 - l'explication reste optionnelle.
        return []

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(values)[::-1][:top_n]
        return [
            {
                "feature": str(feature_names[index]),
                "importance": float(values[index]),
                "explanation": "Variable souvent utilisée par le modèle pour séparer les pannes des heures normales.",
            }
            for index in order
            if values[index] > 0
        ]

    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_[0], dtype=float)
        order = np.argsort(np.abs(coefficients))[::-1][:top_n]
        return [
            {
                "feature": str(feature_names[index]),
                "importance": float(abs(coefficients[index])),
                "coefficient": float(coefficients[index]),
                "direction": "augmente" if coefficients[index] > 0 else "diminue",
                "explanation": (
                    "Coefficient positif : la variable augmente le score de panne. "
                    "Coefficient négatif : elle le diminue."
                ),
            }
            for index in order
            if coefficients[index] != 0
        ]

    return []


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
