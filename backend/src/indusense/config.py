"""Configuration centralisée des pipelines InduSense.

Les chemins sont calculés depuis ce fichier afin que les scripts puissent être
exécutés depuis n'importe quel répertoire de travail.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR
MVTEC_BOTTLE_DIR = DATA_DIR / "mvtec" / "bottle"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
VISION_DATASETS_DIR = ARTIFACTS_DIR / "vision-datasets" / "bottle"
VISION_RUNS_DIR = ARTIFACTS_DIR / "vision-model-runs"
GOLD_DIR = ARTIFACTS_DIR / "gold-datasets"
MAINTENANCE_RUNS_DIR = ARTIFACTS_DIR / "maintenance-ml-runs"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_SEED = 42
COUNTRY_ISO_CODE = "FRA"
