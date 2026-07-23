from __future__ import annotations

import unittest

import numpy as np

from indusense.vision.anomaly import (
    calibrate_threshold,
    image_metrics,
    pixel_error_maps,
    pixel_metrics,
    reconstruction_scores,
)


class VisionAnomalyTests(unittest.TestCase):
    def test_scores_threshold_and_image_metrics(self) -> None:
        images = np.zeros((4, 4, 4, 3), dtype=np.float32)
        reconstructions = images.copy()
        reconstructions[1] += 0.1
        reconstructions[2] += 0.8
        reconstructions[3] += 1.0

        scores = reconstruction_scores(images, reconstructions)
        threshold = calibrate_threshold(scores[:2], percentile=99)
        metrics = image_metrics(np.array([0, 0, 1, 1]), scores, threshold)

        self.assertEqual(scores.shape, (4,))
        self.assertGreater(threshold, scores[0])
        self.assertLess(threshold, scores[2])
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 0, "tp": 2})

    def test_pixel_metrics_use_masks(self) -> None:
        images = np.zeros((2, 2, 2, 3), dtype=np.float32)
        reconstructions = images.copy()
        reconstructions[1, 1, 1, :] = 1.0
        masks = np.zeros((2, 2, 2), dtype=np.float32)
        masks[1, 1, 1] = 1.0

        errors = pixel_error_maps(images, reconstructions)
        metrics = pixel_metrics(masks, errors)

        self.assertEqual(errors.shape, masks.shape)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)

    def test_threshold_rejects_empty_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "non vide"):
            calibrate_threshold(np.array([]))


if __name__ == "__main__":
    unittest.main()
