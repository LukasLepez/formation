from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from indusense.maintenance.model_card import generate_maintenance_model_card


class MaintenanceModelCardTests(unittest.TestCase):
    def test_generates_complete_hugging_face_model_card_from_report(self) -> None:
        report = {
            "run_id": "maintenance_ml_test",
            "created_at": "2026-07-23T08:00:00+00:00",
            "gold_run_name": "gold_test",
            "label_column": "label_failure_next_24h",
            "rows": 1_000,
            "features": 12,
            "class_balance": {"train_positive_rate": 0.08},
            "selected_models": ["logistic_regression", "random_forest"],
            "threshold_strategy": "target_recall",
            "false_negative_cost": 20,
            "false_positive_cost": 1,
            "random_state": 42,
            "carbon": {
                "available": True,
                "duration_seconds": 12.5,
                "energy_kwh": 0.00125,
                "emissions_gco2eq": 0.42,
            },
            "reproducibility": {
                "dataset_hash": "abc123",
                "python_version": "3.14",
                "sklearn_version": "1.7",
            },
            "best_model": "logistic_regression",
            "results": [
                {
                    "model": "logistic_regression",
                    "model_path": "maintenance-ml-runs\\maintenance_ml_test\\models\\logistic_regression.pkl",
                    "threshold": 0.42,
                    "pr_auc_validation": 0.72,
                    "pr_auc_test": 0.70,
                    "roc_auc_validation": 0.81,
                    "roc_auc_test": 0.79,
                    "precision_validation": 0.62,
                    "precision_test": 0.60,
                    "recall_validation": 0.84,
                    "recall_test": 0.80,
                    "f1_validation": 0.71,
                    "f1_test": 0.69,
                    "test_confusion_matrix": {"tn": 850, "fp": 70, "fn": 16, "tp": 64},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = generate_maintenance_model_card(report, Path(temporary_directory))
            content = path.read_text(encoding="utf-8")

        self.assertEqual(path.name, "README.md")
        for section in (
            "### Direct Use",
            "### Downstream Use",
            "### Out-of-Scope Use",
            "## Bias, Risks, and Limitations",
            "## Recommendations",
            "## Evaluation",
            "## Environmental Impact",
            "## Model Card Authors and Contact",
        ):
            self.assertIn(section, content)
        self.assertIn("pipeline_tag: tabular-classification", content)
        self.assertIn("0.4200 gCO₂eq", content)
        self.assertIn("les **24 h** à venir", content)
        self.assertIn("maintenance-ml-runs/maintenance_ml_test/models/logistic_regression.pkl", content)
        self.assertNotIn("{{", content)

    def test_documents_unavailable_codecarbon_measure_without_inventing_value(self) -> None:
        report = {
            "best_model": "decision_tree",
            "carbon": {"available": False, "reason": "tracker indisponible"},
            "results": [{"model": "decision_tree", "threshold": 0.5}],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            content = generate_maintenance_model_card(report, Path(temporary_directory)).read_text(encoding="utf-8")

        self.assertIn("Mesure :** indisponible", content)
        self.assertIn("tracker indisponible", content)
        self.assertIn("ne doit pas être utilisé pour déclarer une empreinte carbone chiffrée", content)


if __name__ == "__main__":
    unittest.main()
