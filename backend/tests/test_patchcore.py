from __future__ import annotations

import unittest

import torch

from indusense.vision.patchcore import PatchCoreConfig, _build_coreset


class PatchCoreTests(unittest.TestCase):
    def test_coreset_is_bounded_and_reproducible(self) -> None:
        patches = torch.arange(1_200, dtype=torch.float32).reshape(100, 12)
        first = _build_coreset(torch, patches, ratio=0.5, max_memory_patches=16, candidate_patches=80, seed=42)
        second = _build_coreset(torch, patches, ratio=0.5, max_memory_patches=16, candidate_patches=80, seed=42)

        self.assertEqual(first.shape, (16, 12))
        self.assertTrue(torch.equal(first, second))

    def test_config_rejects_invalid_memory_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "mémoire"):
            PatchCoreConfig(max_memory_patches=1).validate()


if __name__ == "__main__":
    unittest.main()
