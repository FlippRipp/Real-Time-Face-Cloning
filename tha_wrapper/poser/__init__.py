"""Poser backends: turn (character image, AvatarPose) into an RGBA frame."""

from tha_wrapper.poser.base import PoserBackend
from tha_wrapper.poser.mock import MockPoser

__all__ = ["PoserBackend", "MockPoser", "load_poser_backend"]


def load_poser_backend(config) -> PoserBackend:
    """Create the poser selected by an `AppConfig` (see tha_wrapper.app)."""
    backend = "mock" if config.mock else config.backend
    if backend == "mock":
        return MockPoser(config.char_image_path, size=config.pose_size)
    if backend == "tha3":
        from tha_wrapper.poser.tha3_backend import Tha3Poser

        return Tha3Poser(
            char_image_path=config.char_image_path,
            tha_path=config.tha_path,
            model=config.model,
            device=config.device,
        )
    if backend == "liveportrait":
        from tha_wrapper.poser.liveportrait_backend import LivePortraitPoser

        return LivePortraitPoser(
            char_image_path=config.char_image_path,
            lp_path=config.lp_path,
            cfg=config.lp_cfg,
            python=config.lp_python,
            animal=config.lp_animal,
        )
    raise ValueError(f"unknown poser backend: {backend!r}")
