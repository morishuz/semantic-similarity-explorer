from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import main
from dino_backend import compute_scaled_size


class ResizeTests(unittest.TestCase):
    def test_preserves_aspect_ratio_and_rounds_to_patch_multiple(self):
        width, height = compute_scaled_size(
            1200,
            800,
            target_long_edge=518,
            patch_size=14,
        )

        self.assertEqual(width % 14, 0)
        self.assertEqual(height % 14, 0)
        self.assertGreater(width, height)
        self.assertAlmostEqual(width / height, 1200 / 800, delta=0.08)


class MainCliTests(unittest.TestCase):
    def test_embed_path_uses_extractor_without_error(self):
        with patch.object(main, "DinoFeatureExtractor") as extractor_cls:
            extractor = extractor_cls.return_value
            extractor.image_embedding.return_value = {"mode": "image_embedding"}

            with patch("sys.argv", ["main.py", "embed", str(Path("fixtures/test.ppm"))]):
                main.main()

            extractor.image_embedding.assert_called_once()


if __name__ == "__main__":
    unittest.main()
