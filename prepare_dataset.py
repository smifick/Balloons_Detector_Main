"""Convert the original Matterport Balloon polygons to YOLO detection boxes."""
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "datasets/raw/balloon"
    target = ROOT / "datasets/balloon"
    manifest = {"source": "https://github.com/matterport/Mask_RCNN/releases/tag/v2.1", "splits": {}}
    hashes = {}
    previews = []
    for split in ("train", "val"):
        records = json.loads((source / split / "via_region_data.json").read_text())
        image_dir = target / "images" / split
        label_dir = target / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        files = []
        for record in records.values():
            name = record["filename"]
            image_path = source / split / name
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image: {name}, {hashes[digest]}")
            hashes[digest] = f"{split}/{name}"
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                w, h = image.size
                preview = ImageOps.contain(image.convert("RGB"), (240, 180))
            regions = record["regions"]
            regions = regions.values() if isinstance(regions, dict) else regions
            labels = []
            draw = ImageDraw.Draw(preview)
            for region in regions:
                polygon = region["shape_attributes"]
                xs, ys = polygon["all_points_x"], polygon["all_points_y"]
                x1, x2 = max(0, min(xs)), min(w, max(xs))
                y1, y2 = max(0, min(ys)), min(h, max(ys))
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError(f"Invalid polygon: {name}")
                labels.append(f"0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
                draw.rectangle((x1/w*preview.width, y1/h*preview.height,
                                x2/w*preview.width, y2/h*preview.height),
                               outline="#00ff66", width=2)
            (label_dir / (Path(name).stem + ".txt")).write_text("\n".join(labels) + "\n")
            shutil.copy2(image_path, image_dir / name)
            count += len(labels)
            files.append({"filename": name, "sha256": digest, "objects": len(labels)})
            if len(previews) < 12 and split == "train":
                previews.append(preview)
        manifest["splits"][split] = {"images": len(files), "objects": count, "files": files}
    (target / "data.yaml").write_text(
        "path: " + json.dumps(target.as_posix(), ensure_ascii=False) + "\ntrain: images/train\nval: images/val\nnames:\n  0: balloon\n", encoding="utf-8"
    )
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    canvas = Image.new("RGB", (4 * 240, 3 * 200), "#202020")
    for i, preview in enumerate(previews):
        canvas.paste(preview, ((i % 4) * 240, (i // 4) * 200))
    canvas.save(target / "preview.jpg")
    print(json.dumps({k: {n:v for n,v in s.items() if n != "files"} for k,s in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
