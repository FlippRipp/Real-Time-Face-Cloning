"""TCP control server for the procedural driver.

Protocol: one JSON object per line, both directions.

Request:  {"cmd": "emotion", "name": "happy", "intensity": 1.0}
Response: {"ok": true}  or  {"ok": false, "error": "..."}

Commands (arguments in brackets are optional):

  emotion      name [intensity=1.0] [duration=0.4]
  pose         params={param: value, ...} [duration=0.15]
  head         [pitch] [yaw] [roll] [duration=0.3]
  body         [y] [z] [duration=0.4]
  look         x y [duration=0.15] [head_follow=0.0]
  blink        [double=false]
  speak_visemes  events=[[viseme, duration] | [viseme, duration, weight], ...]
  speak_text   text [seconds_per_syllable=0.18]
  talking      on=true|false
  audio_energy level             (0..1, call repeatedly while audio plays)
  stop_speaking
  idle         [auto_blink] [breathing] [sway] [sway_amount]
  reset        [duration=0.3]
  state                          -> {"ok": true, "state": {...}}
  emotions                       -> {"ok": true, "emotions": [...]}
  help                           -> {"ok": true, "commands": [...]}

The server binds to localhost by default. It applies commands only to the
procedural driver, so in hybrid mode webcam tracking keeps running
underneath whatever the controller does.
"""

from __future__ import annotations

import json
import logging
import socket
import socketserver
import threading
from typing import Any, Dict

from tha_wrapper.drivers.procedural import EMOTIONS, ProceduralDriver

logger = logging.getLogger(__name__)


class ControlServer:
    def __init__(self, driver: ProceduralDriver, host: str = "127.0.0.1",
                 port: int = 9535):
        self.driver = driver
        self.host = host
        self.port = port
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = _make_handler(self)
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self._server = socketserver.ThreadingTCPServer(
            (self.host, self.port), handler
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        logger.info("control server listening on %s:%d", self.host, self.port)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    # ------------------------------------------------------------------

    def handle_command(self, request: Dict[str, Any]) -> Dict[str, Any]:
        cmd = request.get("cmd")
        if not isinstance(cmd, str):
            return {"ok": False, "error": "missing 'cmd'"}
        method = getattr(self, f"_cmd_{cmd}", None)
        if method is None:
            return {"ok": False, "error": f"unknown command {cmd!r}; try 'help'"}
        try:
            result = method(request) or {}
            return {"ok": True, **result}
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    # --- command implementations --------------------------------------

    def _cmd_emotion(self, r):
        self.driver.set_emotion(
            r["name"],
            float(r.get("intensity", 1.0)),
            float(r.get("duration", 0.4)),
        )

    def _cmd_pose(self, r):
        params = r["params"]
        if not isinstance(params, dict):
            raise ValueError("'params' must be an object of {param: value}")
        self.driver.set_params(
            {k: float(v) for k, v in params.items()},
            float(r.get("duration", 0.15)),
        )

    def _cmd_head(self, r):
        self.driver.set_head(
            pitch=_opt_float(r, "pitch"),
            yaw=_opt_float(r, "yaw"),
            roll=_opt_float(r, "roll"),
            duration=float(r.get("duration", 0.3)),
        )

    def _cmd_body(self, r):
        self.driver.set_body(
            y=_opt_float(r, "y"),
            z=_opt_float(r, "z"),
            duration=float(r.get("duration", 0.4)),
        )

    def _cmd_look(self, r):
        self.driver.look_at(
            float(r["x"]),
            float(r["y"]),
            duration=float(r.get("duration", 0.15)),
            head_follow=float(r.get("head_follow", 0.0)),
        )

    def _cmd_blink(self, r):
        self.driver.blink(double=bool(r.get("double", False)))

    def _cmd_speak_visemes(self, r):
        self.driver.speak_visemes([tuple(e) for e in r["events"]])

    def _cmd_speak_text(self, r):
        duration = self.driver.speak_text(
            str(r["text"]), float(r.get("seconds_per_syllable", 0.18))
        )
        return {"duration": duration}

    def _cmd_talking(self, r):
        self.driver.set_talking(bool(r.get("on", True)))

    def _cmd_audio_energy(self, r):
        self.driver.push_audio_energy(float(r["level"]))

    def _cmd_stop_speaking(self, r):
        self.driver.stop_speaking()

    def _cmd_idle(self, r):
        if "auto_blink" in r:
            self.driver.auto_blink = bool(r["auto_blink"])
        if "breathing" in r:
            self.driver.breathing = bool(r["breathing"])
        if "sway" in r:
            self.driver.idle_sway = bool(r["sway"])
        if "sway_amount" in r:
            self.driver.idle_sway_amount = float(r["sway_amount"])

    def _cmd_reset(self, r):
        self.driver.reset(float(r.get("duration", 0.3)))

    def _cmd_state(self, r):
        return {"state": self.driver.state()}

    def _cmd_emotions(self, r):
        return {"emotions": sorted(EMOTIONS)}

    def _cmd_help(self, r):
        return {
            "commands": sorted(
                name[len("_cmd_"):] for name in dir(self)
                if name.startswith("_cmd_")
            )
        }


def _opt_float(r: Dict[str, Any], key: str):
    return float(r[key]) if key in r and r[key] is not None else None


def _make_handler(server: ControlServer):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            logger.info("controller connected: %s", self.client_address)
            for line in self.rfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                    response = server.handle_command(request)
                except json.JSONDecodeError as exc:
                    response = {"ok": False, "error": f"bad JSON: {exc}"}
                except Exception:  # keep the channel alive no matter what
                    logger.exception("error handling control command")
                    response = {"ok": False, "error": "internal error"}
                try:
                    self.wfile.write(
                        (json.dumps(response) + "\n").encode("utf-8")
                    )
                except (BrokenPipeError, ConnectionResetError, socket.error):
                    break
            logger.info("controller disconnected: %s", self.client_address)

    return Handler
