from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from indusense.vision.model_card import generate_vision_model_card


class VisionModelCardTests(unittest.TestCase):
    def test_autoencoder_card_contains_all_b8_sections_and_real_metrics(self) -> None:
        report = {
            "run_id": "vision_ae_test",
            "model_type": "autoencoder",
            "dataset_version": "bottle-test",
            "dataset_hash": "abc123",
            "config": {"loss_name": "mse", "threshold_percentile": 99, "random_seed": 42},
            "architecture": {"input_shape": [256, 256, 3], "latent_shape": [32, 32, 16]},
            "training": {
                "train_normal_images": 167,
                "validation_normal_images": 42,
                "test_images": 64,
                "epochs_completed": 4,
                "best_epoch": 3,
                "augmentation": "images saines uniquement",
            },
            "threshold": {"value": 0.0012, "percentile": 99},
            "metrics": {
                "image": {
                    "auroc": 0.75,
                    "average_precision": 0.8,
                    "precision": 0.7,
                    "recall": 0.6,
                    "f1": 0.64,
                    "confusion_matrix": {"tn": 18, "fp": 2, "fn": 16, "tp": 28},
                },
                "pixel": {"auroc": 0.7, "average_precision": 0.4},
            },
            "carbon": {
                "available": True,
                "country_iso_code": "FRA",
                "duration_seconds": 10,
                "energy_kwh": 0.001,
                "emissions_gco2eq": 0.05,
            },
            "artifacts": {"model": "autoencoder.keras"},
            "critical_analysis": "Validation humaine indispensable.",
            "reproducibility": {"python": "3.14", "keras": "3.15", "backend": "torch"},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            content = generate_vision_model_card(report, Path(temporary_directory)).read_text(encoding="utf-8")

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
        self.assertIn("AUROC image | 0.750000", content)
        self.assertIn("Seuil retenu | 0.00120000", content)
        self.assertIn("exclusivement le split `test`", content)
        self.assertIn("0.0500 gCO₂eq", content)
        self.assertIn("16 défauts manqués", content)


if __name__ == "__main__":
    unittest.main()
