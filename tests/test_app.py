from __future__ import annotations

import unittest

import torch

from app import compute_similarity_map


class ComputeSimilarityMapTests(unittest.TestCase):
    def test_picks_expected_patch_and_normalizes(self):
        left = torch.zeros((2, 2, 3), dtype=torch.float32)
        right = torch.zeros((2, 2, 3), dtype=torch.float32)

        left[1, 0] = torch.tensor([1.0, 0.0, 0.0])
        right[0, 1] = torch.tensor([1.0, 0.0, 0.0])
        right[1, 1] = torch.tensor([0.5, 0.0, 0.0])

        patch_y, patch_x, raw, normalized = compute_similarity_map(
            left,
            right,
            x_norm=0.1,
            y_norm=0.9,
        )

        self.assertEqual((patch_y, patch_x), (1, 0))
        self.assertAlmostEqual(float(raw[0, 1].item()), 1.0)
        self.assertAlmostEqual(float(normalized.max().item()), 1.0)
        self.assertAlmostEqual(float(normalized.min().item()), 0.0)
