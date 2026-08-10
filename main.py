#!/usr/bin/env python3
"""CLI entry point for the THA3 avatar wrapper.

Examples:

  # Procedural avatar, mock renderer (no GPU / THA3 needed), preview window:
  python main.py --mock

  # Webcam-driven VTuber with the real THA3 models, output to OBS:
  python main.py --char images/character.png --driver webcam \
      --tha-path ../talking-head-anime-3-demo --output window,virtualcam

  # AI-driven avatar: procedural driver + control server, virtual cam only:
  python main.py --char images/character.png --driver procedural \
      --tha-path ../talking-head-anime-3-demo --output virtualcam

  # Hybrid: webcam tracking, but AI/hotkeys can override expressions:
  python main.py --char images/character.png --driver hybrid \
      --tha-path ../talking-head-anime-3-demo
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
    parser.add_argument("--tha-path", default="../talking-head-anime-3-demo",
                        help="path to a talking-head-anime-3-demo checkout")
    parser.add_argument("--model", choices=MODEL_NAMES,
                        default="separable_float")
    parser.add_argument("--device", default="cuda",
                        help="torch device (cuda, cuda:0, cpu, mps)")
    parser.add_argument("--mock", action="store_true",
                        help="use the mock poser (no THA3/torch required)")
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

    if not args.mock and not args.char_image_path:
        parser.error("--char is required unless --mock is used")

    return AppConfig(
        char_image_path=args.char_image_path,
        driver=args.driver,
        camera_index=args.camera_index,
        mock=args.mock,
        tha_path=args.tha_path,
        model=args.model,
        device=args.device,
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
