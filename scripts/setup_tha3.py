#!/usr/bin/env python3
"""Set up the THA3 dependencies: PyTorch, the THA3 repo, and its models.

Called by setup.bat, but works standalone on any OS:

    python scripts/setup_tha3.py [--tha-dir DIR] [--torch auto|cpu|skip]
                                 [--skip-models]

Steps (each skipped automatically when already done):
  1. Install PyTorch — tries CUDA wheels when an NVIDIA GPU is present,
     falls back through older CUDA versions, then CPU.
  2. Shallow-clone talking-head-anime-3-demo next to this repository.
  3. Download the THA3 model weights (~800 MB zip, official Dropbox link
     from the THA3 README) and unpack them into <tha-dir>/data/models.
  4. Verify every expected model file exists.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_THA_DIR = os.path.join(os.path.dirname(REPO_ROOT),
                               "talking-head-anime-3-demo")
THA_GIT_URL = "https://github.com/pkhungurn/talking-head-anime-3-demo"

# Official model archive from the THA3 README (CC-BY 4.0, by Pramook
# Khungurn). dl=1 makes Dropbox serve the file directly.
MODELS_URL = ("https://www.dropbox.com/s/y7b8jl4n2euv8xe/"
              "talking-head-anime-3-models.zip?dl=1")

MODEL_VARIANTS = ("separable_float", "separable_half",
                  "standard_float", "standard_half")
MODEL_FILES = ("editor.pt", "eyebrow_decomposer.pt",
               "eyebrow_morphing_combiner.pt", "face_morpher.pt",
               "two_algo_face_body_rotator.pt")

# Newest first; the installer walks down until one succeeds.
CUDA_WHEEL_TAGS = ("cu128", "cu126", "cu124", "cu121")


def info(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def run(cmd, **kwargs) -> int:
    info("$ " + " ".join(cmd))
    return subprocess.call(cmd, **kwargs)


# ----------------------------------------------------------------------
# Step 1: PyTorch
# ----------------------------------------------------------------------

def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.call(
        ["nvidia-smi", "-L"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) == 0


def install_torch(mode: str) -> None:
    if mode == "skip":
        info("skipping PyTorch install (--torch skip)")
        return
    try:
        import torch  # noqa: F401

        info(f"PyTorch already installed (torch {torch.__version__}, "
             f"cuda available: {torch.cuda.is_available()})")
        return
    except ImportError:
        pass

    pip = [sys.executable, "-m", "pip", "install", "torch"]
    if mode == "auto" and has_nvidia_gpu():
        info("NVIDIA GPU detected - installing CUDA build of PyTorch")
        for tag in CUDA_WHEEL_TAGS:
            url = f"https://download.pytorch.org/whl/{tag}"
            if run(pip + ["--index-url", url]) == 0:
                return
            info(f"{tag} wheels unavailable for this Python, trying older...")
        info("falling back to the default PyPI build")
    elif mode == "auto":
        info("no NVIDIA GPU detected - installing CPU PyTorch "
             "(THA3 will be slow; consider --mock mode for testing)")
    if run(pip) != 0:
        raise SystemExit("PyTorch installation failed")


# ----------------------------------------------------------------------
# Step 2: THA3 repository
# ----------------------------------------------------------------------

def clone_tha3(tha_dir: str) -> None:
    if os.path.isdir(os.path.join(tha_dir, "tha3")):
        info(f"THA3 repo already present at {tha_dir}")
        return
    if shutil.which("git") is None:
        raise SystemExit("git is not installed or not on PATH - "
                         "install it from https://git-scm.com/download/win")
    if run(["git", "clone", "--depth", "1", THA_GIT_URL, tha_dir]) != 0:
        raise SystemExit(f"cloning {THA_GIT_URL} failed")


# ----------------------------------------------------------------------
# Step 3: model weights
# ----------------------------------------------------------------------

def missing_model_files(tha_dir: str) -> list:
    models_dir = os.path.join(tha_dir, "data", "models")
    return [
        os.path.join(variant, name)
        for variant in MODEL_VARIANTS for name in MODEL_FILES
        if not os.path.isfile(os.path.join(models_dir, variant, name))
    ]


def download_with_progress(url: str, dest: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[setup] downloading models: "
                      f"{done / (1 << 20):.0f}/{total / (1 << 20):.0f} MB "
                      f"({100 * done // total}%)", end="", flush=True)
            else:
                print(f"\r[setup] downloading models: "
                      f"{done / (1 << 20):.0f} MB", end="", flush=True)
    print(flush=True)


def install_models(tha_dir: str, url: str) -> None:
    if not missing_model_files(tha_dir):
        info("model weights already present")
        return
    models_dir = os.path.join(tha_dir, "data", "models")
    os.makedirs(models_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tha3_models_") as tmp:
        zip_path = os.path.join(tmp, "models.zip")
        info(f"downloading {url}")
        download_with_progress(url, zip_path)
        info("extracting...")
        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        # The archive may nest the variant folders under a top-level dir;
        # find each variant wherever it landed and move it into place.
        for variant in MODEL_VARIANTS:
            target = os.path.join(models_dir, variant)
            if os.path.isdir(target) and not any(
                v.startswith(variant) for v in missing_model_files(tha_dir)
            ):
                continue
            for root, dirs, _files in os.walk(extract_dir):
                if variant in dirs:
                    shutil.rmtree(target, ignore_errors=True)
                    shutil.move(os.path.join(root, variant), target)
                    break
            else:
                raise SystemExit(
                    f"could not find {variant}/ inside the downloaded archive"
                )


# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tha-dir", default=DEFAULT_THA_DIR,
                        help="where to clone talking-head-anime-3-demo")
    parser.add_argument("--torch", default="auto",
                        choices=["auto", "cpu", "skip"],
                        help="auto = CUDA if an NVIDIA GPU is found")
    parser.add_argument("--skip-models", action="store_true",
                        help="do not download the model weights")
    parser.add_argument("--models-url", default=MODELS_URL,
                        help="alternative URL for the models zip")
    args = parser.parse_args()
    tha_dir = os.path.abspath(args.tha_dir)

    install_torch(args.torch)
    clone_tha3(tha_dir)
    if not args.skip_models:
        install_models(tha_dir, args.models_url)
        missing = missing_model_files(tha_dir)
        if missing:
            raise SystemExit(
                "model files still missing after extraction:\n  "
                + "\n  ".join(missing)
            )
        info("all model files verified")

    sample = os.path.join(tha_dir, "data", "images", "lambda_00.png")
    print()
    info("setup complete!")
    info(f"THA3 checkout: {tha_dir}")
    info("try it out:")
    info(f'  python main.py --char "{sample}" --driver procedural '
         f'--tha-path "{tha_dir}"')
    info("(the bundled crypko_*/lambda_* sample images are CC-BY-NC - "
         "non-commercial use only; use your own character for streaming)")


if __name__ == "__main__":
    main()
