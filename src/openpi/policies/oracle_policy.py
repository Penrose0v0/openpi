import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_oracle_example() -> dict:
    """Random input example for the Franka Oracle (single-camera) policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OracleInputs(transforms.DataTransformFn):
    """Input transform for the Franka Oracle dataset.

    State: 9-dim absolute joint angles (7 joints + 2 fingers) — same space as action.
    Image: single third-person view at cam_30deg, both wrist slots zero-padded.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        zeros = np.zeros_like(base_image)

        # PI0_FAST does not mask padded images; PI0/PI05 do.
        wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zeros,
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": wrist_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OracleOutputs(transforms.DataTransformFn):
    """Output transform: keep only the first 9 action dims (7 joints + 2 fingers)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :9])}
