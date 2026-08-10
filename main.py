#!/usr/bin/env python3
"""CLI entry point for the avatar wrapper.

Examples:

  # Procedural avatar, mock renderer (no GPU / models needed), preview window:
  python main.py --mock

  # LivePortrait-rendered avatar (default backend), procedural driver + control
  # server, preview window + OBS virtual camera:
  python main.py --char characters/char.png --output window,virtualcam

  # Same but with the legacy THA3 renderer:
  python main.py --char characters/char.png --backend tha3 \
      --tha-path ../talking-head-anime-3-demo

  # Hybrid: webcam tracking, but AI/hotkeys can override expressions:
  python main.py --char characters/char.png --driver hybrid
"""

import argparse
import logging

from tha_wrapper.app import AppConfig, AvatarApp
from tha_wrapper.poser.tha3_backend import MODEL_NAMES


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Real-time Talking Head Anime 3 avatar with webcam, "
        "procedural/AI control, and OBS virtual camera output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--char", dest="char_image_path", metavar="IMAGE",
                        help="character image (512x512 RGBA, THA3 framing)")
    parser.add_argument("--driver", choices=["webcam", "procedural", "hybrid"],
                        default="procedural")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="webcam device index")
    parser.add_argument("--backend", choices=["liveportrait", "tha3", "mock"],
                        default="liveportrait",
                        help="poser backend (renderer)")
    parser.add_argument("--lp-path",
                        default="../FasterLivePortrait/FasterLivePortrait-windows",
                        help="path to a FasterLivePortrait checkout with "
                        "built TensorRT engines and its bundled venv")
    parser.add_argument("--lp-cfg", default="configs/trt_infer.yaml",
                        help="FasterLivePortrait inference config, relative "
                        "to --lp-path")
    parser.add_argument("--lp-python", default=None,
                        help="python.exe to run the LivePortrait worker "
                        "(default: <lp-path>/venv/python.exe)")
    parser.add_argument("--lp-animal", action="store_true",
                        help="use LivePortrait's animal models (pose only; "
                        "eye/mouth retargeting is disabled)")
    parser.add_argument("--tha-path", default="../talking-head-anime-3-demo",
                        help="path to a talking-head-anime-3-demo checkout")
    parser.add_argument("--model", choices=MODEL_NAMES,
                        default="separable_float",
                        help="THA3 model variant (tha3 backend)")
    parser.add_argument("--device", default="cuda",
                        help="torch device for the tha3 backend")
    parser.add_argument("--mock", action="store_true",
                        help="use the mock poser (no GPU/models required); "
                        "same as --backend mock")
    parser.add_argument("--output", default="window",
                        help="comma-separated: window,virtualcam")
    parser.add_argument("--output-size", type=int, default=512,
                        help="output frame side length in px")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--background", default="green",
                        help="green|magenta|black|white|gray or R,G,B")
    parser.add_argument("--virtualcam-device", default=None,
                        help="explicit virtual camera device (e.g. /dev/video2)")
    parser.add_argument("--no-control", action="store_true",
                        help="disable the TCP control server")
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=9535)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    background = args.background
    if "," in background:
        background = tuple(int(x) for x in background.split(","))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if not args.mock and args.backend != "mock" and not args.char_image_path:
        parser.error("--char is required unless the mock backend is used")

    return AppConfig(
        char_image_path=args.char_image_path,
        driver=args.driver,
        camera_index=args.camera_index,
        backend=args.backend,
        mock=args.mock,
        tha_path=args.tha_path,
        model=args.model,
        device=args.device,
        lp_path=args.lp_path,
        lp_cfg=args.lp_cfg,
        lp_python=args.lp_python,
        lp_animal=args.lp_animal,
        outputs=[o.strip() for o in args.output.split(",") if o.strip()],
        output_size=args.output_size,
        fps=args.fps,
        background=background,
        virtualcam_device=args.virtualcam_device,
        control=not args.no_control,
        control_host=args.control_host,
        control_port=args.control_port,
    )


def main() -> None:
    AvatarApp(parse_args()).run()


if __name__ == "__main__":
    main()
