from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from indusense.vision_dataset import (
    VisionPreparationConfig,
    augment_training_image,
    load_model_input,
    prepare_vision_dataset,
    preprocess_image,
)


class VisionDatasetPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_dir = self.root / "bottle"
        self.artifacts_dir = self.root / "artifacts"
        for relative in (
            "train/good",
            "test/good",
            "test/scratch",
            "ground_truth/scratch",
        ):
            (self.dataset_dir / relative).mkdir(parents=True)

        for index in range(10):
            self._image(f"train/good/{index:03}.png", (20, 10), (20 + index, 80, 140))
        for index in range(2):
            self._image(f"test/good/{index:03}.png", (20, 10), (40, 100 + index, 160))
            self._image(f"test/scratch/{index:03}.png", (20, 10), (180, 40 + index, 30))
            self._image(f"ground_truth/scratch/{index:03}_mask.png", (20, 10), (255, 255, 255))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_manifest_is_deterministic_and_respects_split_boundaries(self) -> None:
        config = VisionPreparationConfig(target_size=64, validation_ratio=0.2, random_seed=42)

        first = prepare_vision_dataset(self.dataset_dir, self.artifacts_dir, config)
        second = prepare_vision_dataset(self.dataset_dir, self.artifacts_dir, config)

        self.assertEqual(first["version_id"], second["version_id"])
        self.assertEqual(first["dataset_hash"], second["dataset_hash"])
        self.assertEqual(first["split_counts"], {"test": 3, "train": 8, "validation": 3})
        self.assertEqual(first["normalization"]["image_count"], 8)
        self.assertFalse(first["leakage_check"]["cross_split_duplicates"])
        self.assertEqual(first["augmentation_policy"]["scope"], "train_only")
        self.assertFalse(first["augmentation_policy"]["vertical_flip"])
        self.assertTrue(all(record["label"] == "good" for record in first["images"] if record["split"] == "train"))
        validation_records = [record for record in first["images"] if record["split"] == "validation"]
        self.assertTrue(any(record["is_anomaly"] for record in validation_records))
        self.assertTrue(any(not record["is_anomaly"] for record in validation_records))
        self.assertTrue(all(record["mask_path"] for record in first["images"] if record["is_anomaly"]))

    def test_letterbox_preserves_ratio_and_adds_padding(self) -> None:
        source = Image.new("RGB", (20, 10), color=(255, 0, 0))

        prepared = preprocess_image(source, VisionPreparationConfig(target_size=64))

        self.assertEqual(prepared.size, (64, 64))
        self.assertEqual(prepared.getpixel((32, 0)), (0, 0, 0))
        self.assertEqual(prepared.getpixel((32, 32)), (255, 0, 0))

    def test_augmentation_is_seeded_and_keeps_dimensions(self) -> None:
        source = Image.new("RGB", (64, 64), color=(90, 120, 150))

        first = augment_training_image(source, seed=7)
        second = augment_training_image(source, seed=7)

        self.assertEqual(first.size, source.size)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_model_input_is_float32_scaled_and_blocks_test_augmentation(self) -> None:
        config = VisionPreparationConfig(target_size=64)
        train_record = {"path": "train/good/000.png", "split": "train", "is_anomaly": False}
        test_record = {"path": "test/scratch/000.png", "split": "test", "is_anomaly": True}

        values = load_model_input(self.dataset_dir, train_record, config, augment=True, seed=3)

        self.assertEqual(values.shape, (64, 64, 3))
        self.assertEqual(values.dtype.name, "float32")
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)
        with self.assertRaisesRegex(ValueError, "réservée aux images normales"):
            load_model_input(self.dataset_dir, test_record, config, augment=True)

    def test_duplicate_content_across_splits_is_rejected(self) -> None:
        duplicate = (self.dataset_dir / "train/good/000.png").read_bytes()
        (self.dataset_dir / "test/good/duplicate.png").write_bytes(duplicate)

        with self.assertRaisesRegex(ValueError, "Fuite de données"):
            prepare_vision_dataset(
                self.dataset_dir,
                self.artifacts_dir,
                VisionPreparationConfig(target_size=64),
            )

    def _image(self, relative_path: str, size: tuple[int, int], color: tuple[int, int, int]) -> None:
        Image.new("RGB", size, color=color).save(self.dataset_dir / relative_path)


if __name__ == "__main__":
    unittest.main()
