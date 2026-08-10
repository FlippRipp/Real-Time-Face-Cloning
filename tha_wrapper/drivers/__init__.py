"""Pose drivers: things that decide what pose the avatar is in each frame."""

from tha_wrapper.drivers.base import PoseDriver
from tha_wrapper.drivers.procedural import ProceduralDriver
from tha_wrapper.drivers.hybrid import HybridDriver

__all__ = ["PoseDriver", "ProceduralDriver", "HybridDriver", "create_driver"]


def create_driver(config):
    """Create the driver selected by an `AppConfig` (see tha_wrapper.app).

    Returns (driver, procedural_driver_or_None). The procedural driver is
    returned separately even in hybrid mode so the control server can be
    attached to it.
    """
    procedural = None
    if config.driver in ("procedural", "hybrid"):
        procedural = ProceduralDriver()
    if config.driver == "procedural":
        return procedural, procedural
    from tha_wrapper.drivers.webcam import WebcamDriver

    webcam = WebcamDriver(camera_index=config.camera_index)
    if config.driver == "webcam":
        return webcam, None
    if config.driver == "hybrid":
        return HybridDriver(webcam, procedural), procedural
    raise ValueError(f"unknown driver: {config.driver!r}")
