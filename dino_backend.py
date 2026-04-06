from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


DEFAULT_MODEL = "facebook/dinov2-base"
TARGET_LONG_EDGE = 518


@dataclass
class DenseFeatures:
    image_path: Path
    model_name: str
    device: str
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    patch_size: int
    patch_grid: torch.Tensor
    cls_embedding: torch.Tensor


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, p=2, dim=-1)


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def compute_scaled_size(
    original_width: int,
    original_height: int,
    *,
    target_long_edge: int,
    patch_size: int,
) -> tuple[int, int]:
    scale = target_long_edge / max(original_width, original_height)
    scaled_width = round_to_multiple(original_width * scale, patch_size)
    scaled_height = round_to_multiple(original_height * scale, patch_size)
    return scaled_width, scaled_height


class DinoFeatureExtractor:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: torch.device | None = None,
    ):
        self.model_name = model_name
        self.device = device or get_device()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._dense_cache: dict[Path, DenseFeatures] = {}

    def _load_model(self):
        if self._processor is None or self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name)
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)

        return self._processor, self._model

    def image_embedding(self, image_path: Path) -> dict[str, object]:
        processor, model = self._load_model()
        image = load_image(image_path)
        inputs = self._preprocess_image(image, processor, model.config)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        embedding = l2_normalize(outputs.pooler_output)[0].detach().cpu()
        return {
            "mode": "image_embedding",
            "image_path": str(image_path),
            "model": self.model_name,
            "device": str(self.device),
            "shape": list(embedding.shape),
            "embedding": embedding.tolist(),
        }

    def dense_features(self, image_path: Path) -> DenseFeatures:
        image_path = image_path.resolve()
        cached = self._dense_cache.get(image_path)
        if cached is not None:
            return cached

        processor, model = self._load_model()
        image = load_image(image_path)
        original_width, original_height = image.size

        inputs = self._preprocess_image(image, processor, model.config)
        batch = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            outputs = model(pixel_values=batch)

        _, _, processed_height, processed_width = batch.shape
        cls_token, patch_grid, patch_size = self._extract_patch_grid(
            outputs.last_hidden_state,
            model.config,
            processed_height=int(processed_height),
            processed_width=int(processed_width),
        )
        dense = DenseFeatures(
            image_path=image_path,
            model_name=self.model_name,
            device=str(self.device),
            original_width=original_width,
            original_height=original_height,
            processed_width=int(processed_width),
            processed_height=int(processed_height),
            patch_size=int(patch_size),
            patch_grid=patch_grid.detach().cpu(),
            cls_embedding=cls_token,
        )
        self._dense_cache[image_path] = dense
        return dense

    def _extract_patch_grid(
        self,
        hidden: torch.Tensor,
        config,
        processed_height: int,
        processed_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        patch_size = config.patch_size
        num_register_tokens = getattr(config, "num_register_tokens", 0)
        patches_h = processed_height // patch_size
        patches_w = processed_width // patch_size
        cls_token = l2_normalize(hidden[:, 0, :])[0].detach().cpu()
        patch_tokens = hidden[:, 1 + num_register_tokens :, :]
        patch_grid = l2_normalize(patch_tokens).unflatten(1, (patches_h, patches_w))[0]
        return cls_token, patch_grid.detach().cpu(), int(patch_size)

    def _preprocess_image(self, image: Image.Image, processor, config) -> dict[str, torch.Tensor]:
        # We disable center cropping so click positions map cleanly by normalized
        # image coordinates. This first version favors intuitive interaction over
        # exact fidelity to the default classification preprocessing.
        target_width, target_height = compute_scaled_size(
            image.width,
            image.height,
            target_long_edge=TARGET_LONG_EDGE,
            patch_size=config.patch_size,
        )
        return processor(
            images=image,
            return_tensors="pt",
            do_center_crop=False,
            size={"height": target_height, "width": target_width},
        )
