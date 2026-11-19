"""Adapt the generic balloon model to one explicitly annotated local camera scene.

The repeated local frame is intentional train oversampling, not independent data.
Original validation images remain held out. Live scene checks measure calibration,
not generalization to other rooms or unseen balloon positions.
"""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def main():
    import cv2
    from ultralytics import YOLO

    target = ROOT / "datasets/balloon_camera"
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            destination = target / kind / split
            destination.mkdir(parents=True, exist_ok=True)
            for source in (ROOT / "datasets/balloon" / kind / split).iterdir():
                if source.is_file():
                    shutil.copy2(source, destination / source.name)
    # Both balloons confirmed by the user in the inspected 640x480 frame.
    boxes = [(269, 290, 304, 325), (205, 252, 224, 271)]
    source = ROOT / "diagnostics/cap_frame.jpg"
    frame = cv2.imread(str(source))
    if frame is None or frame.shape[:2] != (480, 640):
        raise ValueError("Expected the inspected 640x480 calibration frame")
    label = "".join(f"0 {(x1+x2)/2/640:.8f} {(y1+y2)/2/480:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}\n"
                    for x1,y1,x2,y2 in boxes)
    for i in range(16):
        stem = f"local_scene_oversample_{i:02d}"
        shutil.copy2(source, target / "images/train" / (stem + ".jpg"))
        (target / "labels/train" / (stem + ".txt")).write_text(label)
    (target / "data.yaml").write_text(
        "path: " + json.dumps(target.as_posix(), ensure_ascii=False) +
        "\ntrain: images/train\nval: images/val\nnames:\n  0: balloon\n", encoding="utf-8")
    (target / "local_annotation.json").write_text(json.dumps({
        "source": str(source), "xyxy": boxes, "oversampling": 16,
        "independent_local_images": 1, "notes": "Two user-confirmed balloons are positive; laser cap is background."
    }, indent=2))
    preview = frame.copy()
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(ROOT / "diagnostics/chair_annotation.jpg"), preview)
    model = YOLO(str(ROOT / "runs/balloon_yolov8n/weights/best.pt"))
    model.train(data=str(target / "data.yaml"), epochs=50, imgsz=640, batch=8,
                device=0, workers=0, amp=False, project=str(ROOT / "runs"),
                name="balloon_camera_two_targets", exist_ok=False, patience=0,
                optimizer="AdamW", lr0=0.0005, lrf=0.1, warmup_epochs=2,
                hsv_h=0.25, hsv_s=0.4, hsv_v=0.3, degrees=5, translate=0.1,
                scale=0.4, fliplr=0.5, mosaic=0.5, close_mosaic=10,
                seed=42, deterministic=True, plots=True)
    run = Path(model.trainer.save_dir)
    comparisons = []
    for checkpoint in ("best", "last"):
        candidate = YOLO(str(run / "weights" / (checkpoint + ".pt")))
        prediction = candidate.predict(frame, conf=0.05, device=0, verbose=False)[0]
        boxes = [{"xyxy": b.xyxy[0].tolist(), "confidence": float(b.conf[0])} for b in prediction.boxes]
        comparisons.append({"checkpoint": checkpoint, "boxes": boxes})
        cv2.imwrite(str(ROOT / "diagnostics" / f"chair_adapted_{checkpoint}.jpg"), prediction.plot())
    (ROOT / "diagnostics/adaptation_comparison.json").write_text(json.dumps(comparisons, indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
