"""Remote control channel: newline-delimited JSON over TCP.

Lets an external process (an AI agent, a TTS pipeline, a hotkey daemon)
drive the ProceduralDriver while the avatar app runs.
"""

from tha_wrapper.control.server import ControlServer
from tha_wrapper.control.client import AvatarClient

__all__ = ["ControlServer", "AvatarClient"]
