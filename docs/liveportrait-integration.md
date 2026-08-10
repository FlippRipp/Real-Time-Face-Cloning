# LivePortrait backend: evaluation results and integration design

Status as of 2026-08-10. This documents the backend evaluation that led to
choosing LivePortrait as the replacement for THA3, everything set up outside
this repo, and the agreed integration design for the next session.

## Why we're replacing THA3

Filip judged THA3's rendering quality insufficient. Evaluation results:

- **THA4** (`K:\Python\AI\talking-head-anime-4-demo`, working venv + models +
  a custom per-morph override puppeteer app we added): rejected. Quality on
  characters other than the author's originals is below standard, and each
  character needs ~30 GPU-hours of distillation (CC BY-NC licensed models).
- **FasterLivePortrait** (`K:\Python\AI\FasterLivePortrait\FasterLivePortrait-windows`,
  the extract-and-run Windows package): **selected**. Anime feasibility test
  passed — expression transfer quality is far beyond THA-style fixed morphs
  (genuine smiles with teeth, natural gaze), and the TensorRT pipeline runs
  ~31 FPS at 512x512 on the RTX 4090 (measured offline, media pipeline included).

## State of the FasterLivePortrait checkout

- All 15 TensorRT engines are built (9 human, 6 animal) under `checkpoints/`.
  Rebuild with `scripts\all_onnx2trt.bat` (one-time per GPU; do NOT start or
  stop other CUDA apps mid-build — killing Stable Diffusion mid-build crashed
  it with "illegal memory access" once).
- Local patches we made (not upstream):
  - `src/models/__init__.py`: bundled mediapipe segfaults on import on this
    machine; `MediaPipeFaceModel` import made lazy (PEP 562 `__getattr__`).
    Only the `*_mp_infer.yaml` configs need it.
  - `run.py`: camera opens via DSHOW + MJPG at 1280x720 (default mode was
    960x540); added `r` hotkey in realtime mode to re-baseline the neutral
    pose; `q` quits.
  - `src/pipelines/faster_live_portrait_pipeline.py`: added One-Euro
    smoothing of the realtime (webcam) motion descriptor — pitch/yaw/roll,
    t, exp at (4, 0.3), scale at (1.5, 0.3). Without this the face pulsates
    (the source-video smoothing path is skipped in realtime mode).
  - `live.bat` (added): launcher — `live.bat [src_image] [--animal]`,
    defaults to `photo_test.png`. `launch_camera.bat` also exists (wraps
    upstream `camera.bat` with ffmpeg on PATH; ffmpeg is only needed for
    offline video saving, not realtime).
- Test images in the package root: `photo_test.png` (photoreal sanity check),
  `test_portrait.png` (head crop of this repo's `characters/Test.png`),
  `avatar_portrait.png` + `avatar_face.png` (crops of `characters/Avatar.png`).

## Findings on anime detection (the critical gate)

- The InsightFace/RetinaFace detector accepts *some* anime art: Test.png's
  head-and-shoulders crop passed (subtle nose/mouth hints suffice); the
  kitsune Avatar.png failed in human mode in every variant tried (tight crop,
  brightened).
- `--animal` mode detects the kitsune and runs at full speed, but expression
  transfer is heavily damped (by design of the animal retargeting) — not
  good enough as the primary path.
- Agreed follow-up: **landmark injection** — the human pipeline only fails at
  face detection; everything downstream needs facial landmarks, which can be
  supplied manually/from our own anime-face detection (`prepare.py` has one)
  for the *static source image* (one-time per character). This likely unlocks
  human-mode motion for arbitrarily stylized characters. Patch point:
  `FasterLivePortraitPipeline.prepare_source`.

## Open issues

- End-of-session regression, undiagnosed: launching realtime mode showed no
  preview window and spammed "no face in driving frame" (the window only
  appears after the first detected face). The webcam had worked earlier the
  same evening. Suspects: another process holding the camera, or the
  DSHOW+MJPG switch interacting badly with the camera in some states. A probe
  script exists in-session history; re-diagnose if webcam driving is needed.
  Note the integration plan makes the webcam optional (parameter posing).
- Realtime smoothing strengths (4/0.3, scale 1.5/0.3) are untested by Filip —
  tune when live testing resumes.

## Integration design (agreed, not yet started)

Key insight: LivePortrait's driving signal is not pixels — every driving
frame reduces to a motion descriptor `{pitch, yaw, roll, t, scale, exp}`
(degrees + translation + scale + 21x3 implicit expression deltas). The
package's own pkl-replay and JoyVASA (audio->motion) modes prove pixel-free
driving is supported. Therefore **parameter-based posing is viable** and the
webcam is just one optional producer.

Plan (Filip approved starting with step 1):

1. **`LivePortraitPoser` backend** in `tha_wrapper/poser/` wrapping
   `FasterLivePortraitPipeline` (lazy imports per repo rules; path to the
   package configurable like `--tha-path`). One-time `prepare_source` per
   character, then `AvatarPose -> motion descriptor -> frame` per tick.
   The 45-dim `AvatarPose` remains the universal interface; drivers,
   control server, and outputs stay untouched.
2. **Adapter mapping**: head_x/head_y/neck_z -> pitch/yaw/roll (degrees);
   body sway -> t; breathing -> subtle scale oscillation; eye open/close ->
   `retarget_eye` ratio (stitching_eye model); mouth open -> `retarget_lip`
   ratio. These are designed-in numeric interfaces in the pipeline.
3. **Expression bank** for rich emotions (exp space is learned, not named):
   record driving video/webcam once per emotion, extract and save `exp`
   vectors, blend parametrically at runtime (pipeline code documents lip
   indices [6,12,14,17,19,20] and eye indices [11,13,15,16,18] for partial
   blends). Replaces THA3's fixed morphs with a recordable set.
4. Deferred: accessory layer (fox ears/tails via triangle-mesh morphs — see
   the pending background task), landmark injection for stylized sources,
   alpha/compositing for virtualcam (LivePortrait is RGB, no alpha; paste_back
   works on the full source image).

## Licensing note

LivePortrait code is MIT but the InsightFace detection models are
non-commercial; fine for personal use, revisit if that ever changes.
