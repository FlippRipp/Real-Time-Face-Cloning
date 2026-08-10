"""Python client for the avatar control server.

Use this from an AI agent / TTS pipeline / chatbot process:

    from tha_wrapper.control import AvatarClient

    with AvatarClient() as avatar:
        avatar.emotion("happy")
        avatar.look(0.4, 0.1, head_follow=0.5)
        avatar.speak_text("Hello! I am your avatar today.")

Any other language can speak the protocol directly: connect a TCP socket to
127.0.0.1:9535 and exchange newline-delimited JSON (see control/server.py
for the command list).
"""

from __future__ import annotations

import json
import socket
from typing import Dict, List, Sequence, Tuple


class AvatarClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9535,
                 timeout: float = 5.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._file = self._sock.makefile("rwb")

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._sock.close()

    def __enter__(self) -> "AvatarClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------

    def send(self, cmd: str, **kwargs) -> Dict:
        """Send a raw command and return the server's response dict."""
        request = {"cmd": cmd, **kwargs}
        self._file.write((json.dumps(request) + "\n").encode("utf-8"))
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise ConnectionError("control server closed the connection")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"{cmd} failed: {response.get('error')}")
        return response

    # --- convenience wrappers -----------------------------------------

    def emotion(self, name: str, intensity: float = 1.0,
                duration: float = 0.4) -> None:
        self.send("emotion", name=name, intensity=intensity, duration=duration)

    def pose(self, params: Dict[str, float], duration: float = 0.15) -> None:
        self.send("pose", params=params, duration=duration)

    def head(self, pitch: float | None = None, yaw: float | None = None,
             roll: float | None = None, duration: float = 0.3) -> None:
        self.send("head", pitch=pitch, yaw=yaw, roll=roll, duration=duration)

    def body(self, y: float | None = None, z: float | None = None,
             duration: float = 0.4) -> None:
        self.send("body", y=y, z=z, duration=duration)

    def look(self, x: float, y: float, duration: float = 0.15,
             head_follow: float = 0.0) -> None:
        self.send("look", x=x, y=y, duration=duration, head_follow=head_follow)

    def blink(self, double: bool = False) -> None:
        self.send("blink", double=double)

    def speak_visemes(
        self, events: Sequence[Tuple[str, float] | Tuple[str, float, float]]
    ) -> None:
        self.send("speak_visemes", events=[list(e) for e in events])

    def speak_text(self, text: str, seconds_per_syllable: float = 0.18) -> float:
        return self.send(
            "speak_text", text=text, seconds_per_syllable=seconds_per_syllable
        )["duration"]

    def perform(self, script: Sequence[Dict]) -> float:
        """Play a timed action script (non-blocking); returns its duration.

        See control/server.py's `perform` command for the script format.
        A new script replaces the current one; an empty list cancels.
        """
        return self.send("perform", script=list(script))["duration"]

    def stop_perform(self) -> None:
        """Cancel the current script and speech, holding the pose."""
        self.send("stop_perform")

    def talking(self, on: bool = True) -> None:
        self.send("talking", on=on)

    def audio_energy(self, level: float) -> None:
        self.send("audio_energy", level=level)

    def stop_speaking(self) -> None:
        self.send("stop_speaking")

    def idle(self, **kwargs) -> None:
        """kwargs: auto_blink, breathing, sway, sway_amount."""
        self.send("idle", **kwargs)

    def reset(self, duration: float = 0.3) -> None:
        self.send("reset", duration=duration)

    def state(self) -> Dict:
        return self.send("state")["state"]

    def emotions(self) -> List[str]:
        return self.send("emotions")["emotions"]
