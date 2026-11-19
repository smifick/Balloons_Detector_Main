"""Download the public original archive without needing an account/API key."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"


def main():
    raw = ROOT / "datasets/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "balloon_dataset.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        error = z.testzip()
        if error:
            raise ValueError(f"Corrupt archive entry: {error}")
        for info in z.infolist():
            if not info.filename.startswith("balloon/"):
                continue
            destination = (raw / info.filename).resolve()
            if not destination.is_relative_to(raw.resolve()):
                raise ValueError(f"Unsafe archive entry: {info.filename}")
            z.extract(info, raw)
    provenance = {"url": URL, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "bytes": archive.stat().st_size}
    (raw / "download.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
