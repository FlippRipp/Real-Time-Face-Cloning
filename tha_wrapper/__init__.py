"""tha_wrapper - a real-time wrapper around Talking Head Anime 3 (THA3).

Inspired by EasyVtuber, this package animates a single anime-style character
image in real time and can be driven by:

* a webcam (MediaPipe face tracking), like a classic VTuber setup,
* a procedural animation engine (emotions, visemes, gaze, idle motion) that
  an AI agent can control in-process or over a JSON/TCP control channel,
* or both at once (procedural channels override webcam tracking).

Output goes to a preview window and/or an OBS virtual camera.
"""

from tha_wrapper.pose import AvatarPose, POSE_PARAMETER_NAMES

__version__ = "0.1.0"

__all__ = ["AvatarPose", "POSE_PARAMETER_NAMES", "__version__"]
