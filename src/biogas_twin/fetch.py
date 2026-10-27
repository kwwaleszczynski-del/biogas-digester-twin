"""Download of the open farm-scale digester data (Mendeley Data, CC BY 4.0), checksum verified."""
from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)

MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/gk3f363sfg/files/"
                "128d2bbd-153d-4e99-8430-a49f9c82e03c/file_downloaded")
MENDELEY_SHA256 = "ee88e12be5acd05d0c1ec08bcebdfb4f72f15789c91dfddfc4f96277826adbf3"


def mendeley(dest: Path) -> Path:
    """Fetch and unpack the dataset once (about 12 MB)."""
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "farm_scale_biodigester.zip"
    if not zpath.exists():
        log.info("downloading %s", MENDELEY_URL)
        r = requests.get(MENDELEY_URL, timeout=120)
        r.raise_for_status()
        zpath.write_bytes(r.content)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != MENDELEY_SHA256:
        raise RuntimeError(f"checksum mismatch for {zpath}: {digest}")
    out = dest / "extracted"
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():
                if m.endswith(".xlsx") and ".." not in m:
                    z.extract(m, out)
    return out
