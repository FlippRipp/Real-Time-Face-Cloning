"""The main application: driver -> poser -> outputs, at a fixed frame rate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from tha_wrapper.control import ControlServer
from tha_wrapper.drivers import create_driver
from tha_wrapper.output import composite_background, create_outputs
from tha_wrapper.output.window import WindowOutput
from tha_wrapper.poser import load_poser_backend

logger = logging.getLogger(__name__)

# Hotkey -> emotion for the preview window (keys 1..9).
HOTKEY_EMOTIONS = [
    "happy", "sad", "angry", "surprised", "amused", "smug", "tired", "confused",
]


@dataclass
class AppConfig:
    char_image_path: Optional[str] = None
    # Driver
    driver: str = "procedural"          # webcam | procedural | hybrid
    camera_index: int = 0
    # Poser
    backend: str = "liveportrait"       # liveportrait | tha3 | mock
    mock: bool = False                  # shorthand: forces backend = mock
    tha_path: str = "../talking-head-anime-3-demo"
    model: str = "separable_float"
    device: str = "cuda"
    pose_size: int = 512                # mock poser canvas size
    lp_path: str = "../FasterLivePortrait/FasterLivePortrait-windows"
    lp_cfg: str = "configs/trt_infer.yaml"
    lp_python: Optional[str] = None     # default: <lp_path>/venv/python.exe
    lp_animal: bool = False
    # Output
    outputs: List[str] = field(default_factory=lambda: ["window"])
    output_size: int = 512
    fps: int = 30
    background: str = "green"
    virtualcam_device: Optional[str] = None
    # Control server
    control: bool = True
    control_host: str = "127.0.0.1"
    control_port: int = 9535


class AvatarApp:
    def __init__(self, config: AppConfig):
        self.config = config
        logger.info("loading poser backend...")
        self.poser = load_poser_backend(config)
        self.driver, self.procedural = create_driver(config)
        self.outputs = create_outputs(config)
        self.control_server = None
        if config.control and self.procedural is not None:
            self.control_server = ControlServer(
                self.procedural, config.control_host, config.control_port
            )
        self._window = next(
            (o for o in self.outputs if isinstance(o, WindowOutput)), None
        )

    def run(self) -> None:
        config = self.config
        self.driver.start()
        for output in self.outputs:
            output.start()
        if self.control_server is not None:
            self.control_server.start()

        frame_interval = 1.0 / config.fps
        last_time = time.perf_counter()
        fps_avg = 0.0
        try:
            while True:
                now = time.perf_counter()
                dt = min(now - last_time, 0.1)  # clamp huge stalls
                last_time = now

                pose = self.driver.update(dt)
                frame_rgba = self.poser.pose(pose)
                frame_rgb = composite_background(frame_rgba, config.background)
                for output in self.outputs:
                    output.send(frame_rgb)

                fps_avg = fps_avg * 0.9 + (1.0 / max(dt, 1e-6)) * 0.1
                if self._window is not None:
                    self._window.set_hud(
                        f"{fps_avg:5.1f} fps | {config.driver}"
                    )
                    if not self._handle_key(self._window.last_key):
                        break

                # Pace the loop.
                elapsed = time.perf_counter() - now
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("interrupted")
        finally:
            self.close()

    def _handle_key(self, key: int) -> bool:
        """Returns False to quit."""
        if key in (ord("q"), 27):
            return False
        if key == -1 or key == 255:
            return True
        if key == ord("c") and hasattr(self.driver, "calibrate"):
            self.driver.calibrate()
        elif hasattr(self.driver, "webcam") and key == ord("c"):
            self.driver.webcam.calibrate()
        if self.procedural is not None:
            if key == ord("0"):
                self.procedural.reset()
            elif key == ord(" "):
                self.procedural.blink()
            elif key == ord("t"):
                state = self.procedural.state()
                self.procedural.set_talking(not state["talking"])
            elif ord("1") <= key <= ord("9"):
                index = key - ord("1")
                if index < len(HOTKEY_EMOTIONS):
                    self.procedural.set_emotion(HOTKEY_EMOTIONS[index])
        return True

    def close(self) -> None:
        if self.control_server is not None:
            self.control_server.close()
        for output in self.outputs:
            output.close()
        self.driver.close()
        self.poser.close()
