# Keeper - downloads the OPTIONAL SCUNet weights for the learned denoiser.
#
#   .venv\Scripts\python.exe tools\download_models.py
#
# The file (72 MB) comes from the official KAIR release of the SCUNet authors
# and is accepted only if its SHA-256 matches the published value. It is
# written to models/ (git-ignored) through a temporary file, so an aborted
# download never leaves a broken weights file behind.

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    target = ROOT / config.LEARNED_MODEL_DIR / config.LEARNED_MODEL_FILE
    if target.is_file() and sha256(target) == config.LEARNED_MODEL_SHA256:
        print("Already present and verified: %s" % target)
        return 0

    target.parent.mkdir(exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print("Downloading %s ..." % config.LEARNED_MODEL_URL)
    try:
        with urllib.request.urlopen(config.LEARNED_MODEL_URL, timeout=60) as response, open(partial, "wb") as f:
            shutil.copyfileobj(response, f, 1 << 20)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        print("Download failed: %s" % exc)
        return 1

    if sha256(partial) != config.LEARNED_MODEL_SHA256:
        partial.unlink()
        print("Checksum mismatch - the file was discarded.")
        return 1

    os.replace(partial, target)
    print("Saved and verified: %s" % target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
