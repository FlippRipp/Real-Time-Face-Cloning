"""Frame outputs: preview window and OBS virtual camera."""

from tha_wrapper.output.base import FrameOutput, composite_background

__all__ = ["FrameOutput", "composite_background", "create_outputs"]


def create_outputs(config):
    """Create the outputs selected by an `AppConfig` (see tha_wrapper.app)."""
    outputs = []
    if "window" in config.outputs:
        from tha_wrapper.output.window import WindowOutput

        outputs.append(WindowOutput(title="tha_wrapper"))
    if "virtualcam" in config.outputs:
        from tha_wrapper.output.virtualcam import VirtualCamOutput

        outputs.append(
            VirtualCamOutput(
                width=config.output_size,
                height=config.output_size,
                fps=config.fps,
                device=config.virtualcam_device,
            )
        )
    return outputs
