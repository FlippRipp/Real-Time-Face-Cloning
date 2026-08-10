"""The real THA3 poser backend.

Requires a checkout of https://github.com/pkhungurn/talking-head-anime-3-demo
with its model weights downloaded into `data/models/` (see that repo's
README). Point `--tha-path` at the checkout; it is added to `sys.path` and
the poser is loaded through THA3's own `load_poser` entry point, exactly the
way EasyVtuber wraps THA.

The name -> pose-vector-index mapping is rebuilt from the loaded poser's
parameter groups rather than hardcoded, so this stays correct even if THA3
reorders parameters.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

import numpy as np

from tha_wrapper.pose import POSE_PARAMETER_NAMES, AvatarPose
from tha_wrapper.poser.base import PoserBackend

logger = logging.getLogger(__name__)

MODEL_NAMES = ("standard_float", "separable_float", "standard_half", "separable_half")


class Tha3Poser(PoserBackend):
    def __init__(
        self,
        char_image_path: str,
        tha_path: str,
        model: str = "separable_float",
        device: str = "cuda",
    ):
        if model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}")
        tha_path = os.path.abspath(os.path.expanduser(tha_path))
        if not os.path.isdir(os.path.join(tha_path, "tha3")):
            raise FileNotFoundError(
                f"{tha_path} does not look like a talking-head-anime-3-demo "
                "checkout (no tha3/ package inside). Clone "
                "https://github.com/pkhungurn/talking-head-anime-3-demo and "
                "download its models, then pass --tha-path."
            )
        if tha_path not in sys.path:
            sys.path.insert(0, tha_path)

        import torch
        from tha3.poser.modes.load_poser import load_poser

        self._torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            device = "cpu"
        self._device = torch.device(device)
        self._poser = poser = load_poser(model, device)
        self.size = poser.get_image_size()
        self._dtype = poser.get_dtype()

        self._index_of = self._build_parameter_index(poser)
        self._num_parameters = poser.get_num_parameters()

        self._input_image = self._load_input_image(char_image_path, tha_path)
        # Warm up (first call compiles/allocates and is very slow).
        self.pose(AvatarPose())

    @staticmethod
    def _build_parameter_index(poser) -> Dict[str, int]:
        """Map our parameter names to indices in the poser's pose vector."""
        index_of: Dict[str, int] = {}
        try:
            for group in poser.get_pose_parameter_groups():
                name = group.get_group_name()
                base = group.get_parameter_index()
                if group.get_arity() == 2:
                    index_of[f"{name}_left"] = base
                    index_of[f"{name}_right"] = base + 1
                else:
                    index_of[name] = base
        except Exception:  # pragma: no cover - defensive against THA3 changes
            logger.exception(
                "Could not introspect THA3 pose parameters; "
                "falling back to the canonical ordering"
            )
            index_of = {n: i for i, n in enumerate(POSE_PARAMETER_NAMES)}

        missing = [n for n in POSE_PARAMETER_NAMES if n not in index_of]
        if missing:
            raise RuntimeError(
                f"THA3 poser is missing expected parameters: {missing}. "
                "This wrapper targets THA3 models."
            )
        return index_of

    def _load_input_image(self, char_image_path: str, tha_path: str):
        from tha3.util import (
            extract_pytorch_image_from_PIL_image,
            resize_PIL_image,
        )
        from PIL import Image

        pil_image = Image.open(char_image_path).convert("RGBA")
        pil_image = resize_PIL_image(pil_image, (self.size, self.size))
        torch_image = extract_pytorch_image_from_PIL_image(pil_image)
        return torch_image.to(self._device).to(self._dtype)

    def pose(self, pose: AvatarPose) -> np.ndarray:
        torch = self._torch
        vector = torch.zeros(
            self._num_parameters, dtype=self._dtype, device=self._device
        )
        for name, value in pose.as_dict(sparse=True).items():
            vector[self._index_of[name]] = value

        with torch.inference_mode():
            output = self._poser.pose(self._input_image, vector)[0]

        # (4, H, W) in [-1, 1] -> (H, W, 4) uint8
        output = (output.detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5
        frame = output.permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
        return frame
