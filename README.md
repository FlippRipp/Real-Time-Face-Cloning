# Real-Time Face Cloning — animated avatar wrapper

A real-time avatar engine that animates a single character image and streams it to an
**OBS virtual camera** — driven by your **webcam**, **procedurally by code/an AI agent**,
or **both at once**. Rendering is done by
[FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) (TensorRT, default)
or the legacy [Talking Head Anime 3 (THA3)](https://github.com/pkhungurn/talking-head-anime-3-demo)
backend; the architecture is inspired by [EasyVtuber](https://github.com/GunwooHan/EasyVtuber).

```
                 ┌─────────────────────┐
  webcam ──────► │ WebcamDriver        │──┐
  (MediaPipe)    └─────────────────────┘  │   ┌──────────────┐   ┌──────────────────┐
                                          ├──►│ LivePortrait │──►│ window preview   │
  AI agent ────► ┌─────────────────────┐  │   │ / THA3 poser │   │ OBS virtual cam  │
  TCP/JSON or    │ ProceduralDriver    │──┘   │ (or mock)    │   └──────────────────┘
  in-process     │ emotions · visemes  │      └──────────────┘
                 │ gaze · idle motion  │
                 └─────────────────────┘
```

## Features

- **LivePortrait renderer (default)** — wraps a
  [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) TensorRT pipeline
  for photoreal-quality expression transfer (real smiles with teeth, natural gaze) at
  ~30 FPS on an RTX-class GPU. The pipeline runs in a worker subprocess inside the
  FasterLivePortrait package's own venv (its TensorRT engines are version-locked to it);
  poses stream to the worker as LivePortrait motion descriptors, so no webcam or driving
  video is needed. Point `--lp-path` at the package directory.
- **Webcam driver** — MediaPipe FaceLandmarker face tracking: blinks, mouth shapes, eyebrows,
  iris/gaze, and head pose (solvePnP), with smoothing and press-`c`-to-calibrate.
- **Procedural driver** — animate the avatar entirely from code: emotion presets with
  eased transitions, viseme-timeline lipsync (TTS-ready), `speak_text` placeholder
  lipsync, live audio-energy lipsync, gaze targeting with head-follow, auto-blink,
  breathing, and organic idle sway. Fully thread-safe. Timed `perform` scripts let an
  AI hand over a whole screenplay (emotion shifts, speech, glances, blinks) in one
  non-blocking call; the driver executes the timing.
- **Hybrid driver** — webcam tracking underneath, with procedural channels overriding
  per-parameter and easing back to tracking when released.
- **Control server** — newline-delimited JSON over TCP (`127.0.0.1:9535`), plus a Python
  `AvatarClient`, so an external AI process (chatbot, TTS pipeline, game logic) can drive
  the avatar in any language.
- **Outputs** — OpenCV preview window with hotkeys, and OBS virtual camera via
  `pyvirtualcam` with chroma-key-ready backgrounds.
- **Mock poser** — run and test the whole pipeline (drivers, control channel, outputs)
  without a GPU, PyTorch, or the THA3 models.

## Installation

### Windows one-shot setup

```bat
setup.bat
```

This creates a `.venv`, installs all Python dependencies, installs PyTorch (the CUDA
build if an NVIDIA GPU is detected), clones `talking-head-anime-3-demo` next to this
repo, downloads the THA3 model weights (~800 MB, official link from the THA3 README),
and verifies every model file. Then launch with `run.bat`: with no arguments it shows
an interactive menu (pick a mode, then pick a character from the `characters\` folder
by number); with arguments it forwards them to `main.py`, e.g. `run.bat --mock`.

Options: `setup.bat --torch cpu` (force CPU PyTorch), `setup.bat --skip-models`,
`setup.bat --tha-dir D:\somewhere\tha3`. The underlying helper is cross-platform —
on Linux/macOS run `python scripts/setup_tha3.py` after `pip install -r requirements.txt`.

The only things it can't install for you: OBS Studio (≥ 26) for the virtual camera
driver, and your character art.

### Manual setup

```bash
pip install -r requirements.txt
```

That covers webcam tracking and the virtual camera. For real rendering you need one of
the two backends below.

**LivePortrait (default backend):** get a FasterLivePortrait Windows package (the
extract-and-run build with its bundled `venv\python.exe`), build its TensorRT engines
once for your GPU (`scripts\all_onnx2trt.bat` inside the package), and point `--lp-path`
at the package directory (default: `../FasterLivePortrait/FasterLivePortrait-windows`).
Nothing needs to be installed into this repo's venv — the backend runs the pipeline in a
worker subprocess using the package's own Python. Character images with transparency are
handled automatically (the character's alpha channel is reapplied to the rendered frames).
Note: LivePortrait's face detector must find a face in the character image; for stylized
art that fails detection, use a tighter head-and-shoulders crop for now.

**THA3 (legacy backend, `--backend tha3`):**

1. Install PyTorch matching your CUDA setup: <https://pytorch.org/get-started/locally/>
   (a GPU is strongly recommended; `separable_half` runs lightest).
2. Clone the THA3 demo and download its models per its README:

   ```bash
   git clone https://github.com/pkhungurn/talking-head-anime-3-demo ../talking-head-anime-3-demo
   # then place the model files under ../talking-head-anime-3-demo/data/models/
   ```

3. Prepare a character image: 512×512 RGBA PNG, forward-facing anime-style character with
   transparent background, framed per the THA3 spec (head roughly in the upper half).
   Drop it in the `characters/` folder so `run.bat`'s menu can find it.
   The preparation wizard (`python -m tha_wrapper.prepare art.png`, or run.bat menu
   option 5) converts arbitrary art for you: AI background removal (rembg anime model),
   face-detected framing against the THA3 guides, and a constraint report — every stage
   previewed and adjustable before saving into `characters/`.

For the virtual camera: on Windows/macOS install OBS (≥ 26) for its virtual camera driver;
on Linux load `v4l2loopback` (`sudo modprobe v4l2loopback devices=1`).

## Usage

```bash
# Try the pipeline with no GPU/models at all (placeholder renderer):
python main.py --mock

# Classic VTuber: webcam-driven, preview + OBS virtual camera
# (LivePortrait renderer by default; add --backend tha3 for THA3):
python main.py --char characters/char.png --driver webcam --output window,virtualcam

# AI avatar: procedural driver + control server, virtual camera only:
python main.py --char characters/char.png --driver procedural --output virtualcam

# Hybrid: you drive it with your face, the AI overrides expressions:
python main.py --char characters/char.png --driver hybrid
```

In OBS, add a *Video Capture Device* source and pick the virtual camera; with
`--background green` (default) add a *Chroma Key* filter to make the avatar transparent.

Preview-window hotkeys: `q` quit · `c` calibrate webcam · `1`–`8` emotions ·
`0` reset · `space` blink · `t` toggle talking.

## Driving the avatar from an AI

Run the app with `--driver procedural` (or `hybrid`), then connect to the control server —
newline-delimited JSON over TCP, default `127.0.0.1:9535`:

```python
from tha_wrapper.control import AvatarClient

with AvatarClient() as avatar:
    avatar.emotion("happy", intensity=0.8)
    avatar.look(0.4, 0.1, head_follow=0.5)
    avatar.blink()
    seconds = avatar.speak_text("Hello! I'm being driven by an AI.")
    # or, with real TTS phoneme timings:
    avatar.speak_visemes([("oh", 0.18), ("aa", 0.22), ("oh", 0.18), ("ou", 0.28)])
    # or, while streaming TTS audio, feed loudness for live lipsync:
    avatar.audio_energy(0.7)
```

The highest-level primitive is `perform`: a timed script the driver plays out on its
own clock, so one call carries a whole expressive performance and returns immediately —
ideal for LLM tool calling:

```python
avatar.perform([
    {"emotion": "surprised", "intensity": 1.0, "duration": 0.2},
    {"at": 0.1, "blink": "double"},
    {"at": 0.3, "say": "Oh! I was not expecting that at all."},
    {"at": 0.9, "emotion": "amused", "intensity": 0.6},
    {"at": 1.6, "look": [0.4, -0.1], "head_follow": 0.3},
])
```

From any other language, send one JSON object per line:

```json
{"cmd": "emotion", "name": "surprised", "intensity": 1.0}
{"cmd": "look", "x": -0.5, "y": 0.2, "head_follow": 0.4}
{"cmd": "speak_visemes", "events": [["aa", 0.2], ["ih", 0.15], ["sil", 0.1]]}
{"cmd": "state"}
```

Commands: `emotion`, `pose` (raw THA3 parameters), `head`, `body`, `look`, `blink`,
`speak_visemes`, `speak_text`, `perform`, `stop_perform`, `talking`, `audio_energy`,
`stop_speaking`, `idle`, `reset`, `state`, `emotions`, `help` — full argument reference
in [`tha_wrapper/control/server.py`](tha_wrapper/control/server.py).

Emotion presets: `neutral`, `happy`, `sad`, `angry`, `surprised`, `amused`, `smug`,
`tired`, `confused` (defined in
[`tha_wrapper/drivers/procedural.py`](tha_wrapper/drivers/procedural.py) — easy to extend).
Visemes: `aa`, `ih`, `ou`, `eh`, `oh`, `dd`, `sil`.

Examples:

- [`examples/scripted_performance.py`](examples/scripted_performance.py) — a scripted
  routine exercising emotions, gaze, blinks, and both lipsync paths.
- [`examples/ai_agent_avatar.py`](examples/ai_agent_avatar.py) — a JARVIS-style
  companion: an LLM (any OpenAI-compatible endpoint; defaults to DeepSeek via
  OpenRouter) writes timed stage scripts through a `perform` tool call, and the
  avatar acts them out while the chat stays responsive.

The `ProceduralDriver` can also be used directly in-process (it is thread-safe) if your
agent lives in the same Python program — see `tha_wrapper/drivers/procedural.py`.

## Project layout

```
tha_wrapper/
  pose.py            # named 45-dim THA3 pose model (AvatarPose)
  poser/             # LivePortrait backend (worker subprocess) + THA3 + mock
  drivers/           # webcam (MediaPipe), procedural (AI-drivable), hybrid
  control/           # TCP JSON control server + AvatarClient
  output/            # preview window, OBS virtual camera, compositing
  app.py             # render loop wiring it all together
main.py              # CLI
examples/            # control-channel demos (scripted + LLM-driven)
tests/               # unit tests (run without torch/mediapipe/pyvirtualcam)
```

## Testing

```bash
pip install pytest numpy
python -m pytest tests/
```

The tests (and `--mock` mode) run without any of the heavy optional dependencies.

## Credits

- [pkhungurn/talking-head-anime-3-demo](https://github.com/pkhungurn/talking-head-anime-3-demo) —
  the THA3 models this wraps. Note THA3's own license terms for models and generated output.
- [GunwooHan/EasyVtuber](https://github.com/GunwooHan/EasyVtuber) — the architecture this
  wrapper takes inspiration from (facial capture → THA → virtual camera).
