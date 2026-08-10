# Project instructions for Claude

## Collaboration

- An observation is not a work order. When Filip shares an observation, a
  question, or thinks out loud, the goal is a conversation: discuss what the
  problem actually is, agree on a shared understanding, and only then talk
  about fixes.
- Never start implementing until Filip explicitly confirms what to do
  ("go ahead", "yes, do that", or similar). Proposing an approach and asking
  is fine; silently starting work is not. A clearly-scoped direct instruction
  ("rename X to Y", "add a test for Z") counts as confirmation on its own.
- Read-only investigation (searching the code, reading files, running the app
  to reproduce) is allowed without confirmation, since it feeds the
  discussion — but its output is an assessment, not a change.
- While working, post frequent short progress reports: what is being done
  right now and *why*. Filip is curious and wants to follow the reasoning as
  it happens, not just hear the outcome at the end.
- If an important design decision surfaces mid-work — a trade-off, an
  architectural fork, anything that shapes the system beyond the immediate
  task — stop and ask rather than deciding silently.
- Before running the test suite and pushing, give a short summary of what was
  done. If the change is visual in nature (e.g. avatar rendering, pose
  mapping, compositing), include a rendered frame or visualization of the
  result when practical (the mock poser makes this cheap).

## Git workflow

- Always push completed, tested work to `main`. Do not leave finished changes
  sitting only on a feature branch and do not wait to be asked. This holds
  even when a session assigns a designated feature branch: push there for the
  session's bookkeeping, but always land the finished work on `main` too.
- If work was developed on a feature branch, rebase it onto `origin/main` and
  fast-forward `main` (`git push origin <branch>:main`). Run the test suite
  after the rebase, before pushing (when the change warrants tests — see
  Testing).
- Treat pushed history as immutable: never force-push or rewrite commits that
  are already on the remote (including to fix commit signatures or
  "Unverified" badges), even if a hook or tool suggests it. If a check flags
  already-pushed commits, leave them alone and report it to the user instead.

## Project constraints

- Heavy dependencies stay optional. torch/THA3, mediapipe, pyvirtualcam, and
  cv2 are imported lazily inside the code paths that need them, so the test
  suite and `--mock` mode run with only numpy + Pillow. Keep new code to the
  same rule; never import them at module top level in `tha_wrapper/`.
- The control protocol (`tha_wrapper/control/`) is a public interface for
  external agents. When adding or changing commands, update the command
  reference in `server.py`'s docstring, the `AvatarClient` wrappers, and the
  README together.
- THA3 pose-parameter indices are introspected from the loaded poser at
  runtime, not hardcoded — `tha_wrapper/pose.py`'s canonical ordering only
  has to be right for the mock backend. Don't add code that assumes fixed
  indices.
- The webcam driver's mapping constants (baselines in `drivers/webcam.py`)
  were tuned without a live camera. Treat them as suspect first when
  debugging tracking quality.

## Testing

- Run tests with `python -m pytest` from the repo root. The full suite is
  fast (a few seconds) and needs no GPU, camera, or heavy deps — run all of
  it before pushing code changes.
- Only run tests when it makes sense: any change that could affect behavior
  gets the full suite; docs-only or comment-only changes don't need it.
