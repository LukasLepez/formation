from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from indusense.vision.train import VisionTrainingConfig, train_vision_autoencoder


class VisionTrainingIntegrationTests(unittest.TestCase):
    def test_one_epoch_produces_all_tp_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = root / "bottle"
            run_dir = root / "runs" / "integration"
            records = []
            for index, split in enumerate(("train", "train", "validation")):
                relative = f"train/good/{index:03d}.png"
                self._normal_image(dataset_dir / relative, shade=80 + index * 5)
                records.append({"path": relative, "split": split, "label": "good", "is_anomaly": False, "mask_path": None})

            good_path = "test/good/000.png"
            defect_path = "test/scratch/000.png"
            mask_path = "ground_truth/scratch/000_mask.png"
            self._normal_image(dataset_dir / good_path, shade=90)
            self._defect_image(dataset_dir / defect_path, dataset_dir / mask_path)
            records.extend(
                [
                    {"path": good_path, "split": "test", "label": "good", "is_anomaly": False, "mask_path": None},
                    {"path": defect_path, "split": "test", "label": "scratch", "is_anomaly": True, "mask_path": mask_path},
                ]
            )
            manifest = {
                "version_id": "bottle-integration",
                "dataset_hash": "integration-hash",
                "config": {
                    "target_size": 64,
                    "validation_ratio": 0.2,
                    "defect_validation_ratio": 0.3,
                    "random_seed": 42,
                    "padding_value": 0,
                    "interpolation": "bilinear",
                },
                "images": records,
            }

            report = train_vision_autoencoder(
                dataset_dir=dataset_dir,
                manifest=manifest,
                run_dir=run_dir,
                run_id="integration",
                config=VisionTrainingConfig(epochs=1, batch_size=1, early_stopping_patience=1),
                mlflow_tracking_uri=(root / "mlruns").as_uri(),
                experiment_name="vision_integration_test",
            )

            self.assertEqual(report["status"], "success")
            self.assertEqual(report["training"]["epochs_completed"], 1)
            self.assertEqual(report["architecture"]["compression_ratio"], 12.0)
            self.assertIsNotNone(report["metrics"]["image"]["auroc"])
            for artifact in report["artifacts"].values():
                self.assertTrue((run_dir / artifact).is_file(), artifact)
            self.assertTrue(report["samples"])
            self.assertTrue((run_dir / report["samples"][0]["heatmap_artifact"]).is_file())

    @staticmethod
    def _normal_image(path: Path, shade: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(shade, shade, shade)).save(path)

    @staticmethod
    def _defect_image(path: Path, mask_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (64, 64), color=(90, 90, 90))
        mask = Image.new("L", (64, 64), color=0)
        ImageDraw.Draw(image).rectangle((24, 8, 30, 56), fill=(255, 20, 20))
        ImageDraw.Draw(mask).rectangle((24, 8, 30, 56), fill=255)
        image.save(path)
        mask.save(mask_path)


if __name__ == "__main__":
    unittest.main()
