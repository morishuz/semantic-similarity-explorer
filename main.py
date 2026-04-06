from __future__ import annotations

import argparse
import json
from pathlib import Path

from dino_backend import DEFAULT_MODEL, DinoFeatureExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 image embeddings or dense patch features."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser(
        "embed", help="Extract one normalized embedding vector for an image."
    )
    embed_parser.add_argument("image", type=Path, help="Path to an image file.")

    dense_parser = subparsers.add_parser(
        "dense",
        help="Extract normalized per-patch features for an image.",
    )
    dense_parser.add_argument("image", type=Path, help="Path to an image file.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    extractor = DinoFeatureExtractor(model_name=DEFAULT_MODEL)

    if args.command == "embed":
        result = extractor.image_embedding(args.image)
    else:
        dense = extractor.dense_features(args.image)
        result = {
            "mode": "dense_features",
            "image_path": str(dense.image_path),
            "model": dense.model_name,
            "device": dense.device,
            "original_size": [dense.original_height, dense.original_width],
            "input_resolution": [dense.processed_height, dense.processed_width],
            "patch_size": dense.patch_size,
            "patch_grid_shape": list(dense.patch_grid.shape),
            "cls_embedding_shape": list(dense.cls_embedding.shape),
            "patch_features": dense.patch_grid.tolist(),
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
