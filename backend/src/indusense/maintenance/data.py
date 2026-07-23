"""Chargement et découpage temporel du Gold dataset."""

from pathlib import Path
from typing import Any

import pandas as pd

from indusense.maintenance_ml import resolve_gold_csv, select_feature_columns, validate_gold_for_ml


def load_gold_splits(gold_dir: Path, label_column: str, gold_run_name: str | None = None) -> tuple[Path, dict[str, Any]]:
    path = resolve_gold_csv(gold_dir, gold_run_name)
    frame = pd.read_csv(path)
    frame["window_start"] = pd.to_datetime(frame["window_start"], errors="coerce")
    frame = frame.sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)
    validate_gold_for_ml(frame, label_column)
    features = select_feature_columns(frame, label_column)
    splits = {name: frame.loc[frame["split_set"] == name].copy() for name in ("train", "validation", "test")}
    return path, {"frame": frame, "features": features, "splits": splits}
