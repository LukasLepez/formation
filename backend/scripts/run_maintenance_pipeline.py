"""Pipeline B7 : Gold → baseline → Optuna → CO₂ → SHAP → rapports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indusense import config
from indusense.maintenance_ml import MaintenanceMlConfig, train_maintenance_models


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exécute la pipeline B7 de maintenance prédictive.")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--gold-run", default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--study-mode", choices=("frugal", "heavy"), default="frugal")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    run_id = f"maintenance_b7_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    report = train_maintenance_models(MaintenanceMlConfig(
        gold_dir=config.GOLD_DIR, run_dir=config.MAINTENANCE_RUNS_DIR / run_id,
        label_column=f"label_failure_next_{args.horizon}h", gold_run_name=args.gold_run,
        tune=args.tune, tune_n_trials=args.n_trials, tune_timeout_seconds=args.timeout,
        tune_mode=args.study_mode, random_state=args.seed,
    ))
    print(json.dumps({"run_id": run_id, "best_model": report["best_model"], "report": str(config.MAINTENANCE_RUNS_DIR / run_id / "report.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
