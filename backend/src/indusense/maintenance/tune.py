"""Étude Optuna bornée sur XGBoost avec validation temporelle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

def run_xgboost_study(
    pipeline: Any, x_train: pd.DataFrame, y_train: pd.Series, *, n_trials: int,
    timeout_seconds: int, seed: int, aggressive_pruning: bool, output_dir: Path,
) -> dict[str, Any]:
    """Optimise PR-AUC CV, avec TPE, pruning et budget explicite."""
    try:
        import optuna
        from optuna.importance import get_param_importances
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    except ImportError as error:
        return {"available": False, "reason": f"Optuna indisponible : {error}"}

    output_dir.mkdir(parents=True, exist_ok=True)
    def objective(trial: Any) -> float:
        tuned = clone(pipeline)
        params = {
            "model__n_estimators": trial.suggest_int("n_estimators", 80, 400),
            "model__max_depth": trial.suggest_int("max_depth", 3, 10),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "model__min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "model__scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 30.0),
        }
        tuned.set_params(**params)
        # Import tardif : ce module est utilisé par ``maintenance_ml`` lui-même.
        from indusense.maintenance_ml import temporal_cross_validate
        metrics = temporal_cross_validate(tuned, x_train, y_train)
        value = float(metrics["pr_auc_mean"])
        trial.report(value, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0) if aggressive_pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, catch=(ValueError,))
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed:
        return {"available": True, "status": "no_completed_trial", "trials": len(study.trials)}
    history_path = output_dir / "optuna_history.png"
    importance_path = output_dir / "optuna_importance.png"
    try:
        axis = plot_optimization_history(study); axis.figure.savefig(history_path, dpi=150, bbox_inches="tight")
        axis = plot_param_importances(study); axis.figure.savefig(importance_path, dpi=150, bbox_inches="tight")
    except Exception:
        pass
    importances = {key: float(value) for key, value in get_param_importances(study).items()}
    return {"available": True, "status": "success", "best_value": float(study.best_value), "best_params": study.best_params, "trials": len(study.trials), "completed_trials": len(completed), "pruned_trials": sum(t.state.name == "PRUNED" for t in study.trials), "param_importances": importances, "artifacts": {"history": str(history_path) if history_path.exists() else None, "importance": str(importance_path) if importance_path.exists() else None}}
