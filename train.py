"""Fine-tune YOLOv8n; save best weights and measured validation metrics."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the PyTorch CUDA installation.")
    run = ROOT / "runs/balloon_yolov8n"
    model = YOLO(str(run / "weights/last.pt") if args.resume else str(ROOT / "yolov8n.pt"))
    if args.resume:
        model.train(resume=True)
    else:
        model.train(
            data=str(ROOT / "datasets/balloon/data.yaml"), epochs=args.epochs,
            imgsz=640, batch=args.batch, device=0, workers=0,
            project=str(ROOT / "runs"), name="balloon_yolov8n", exist_ok=False,
            seed=42, deterministic=True, patience=20, amp=False,
            hsv_h=0.5, hsv_s=0.5, hsv_v=0.4,
            degrees=10, translate=0.1, scale=0.5, fliplr=0.5,
            mosaic=1.0, close_mosaic=10, plots=True, save=True,
        )
    run = Path(model.trainer.save_dir)
    best = YOLO(str(run / "weights/best.pt"))
    metrics = best.val(data=str(ROOT / "datasets/balloon/data.yaml"), device=0,
                       workers=0, plots=True, project=str(ROOT / "runs"), name="validation")
    with (run / "results.csv").open(encoding="utf-8") as handle:
        epochs = list(csv.DictReader(handle))
    best_epoch = max(epochs, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    report = {"metrics": {k: float(v) for k,v in metrics.results_dict.items()},
              "epochs_completed": len(epochs), "best_epoch": int(best_epoch["epoch"]),
              "speed_ms": metrics.speed, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "weights": str(run / "weights/best.pt"),
              "limitations": "13-image validation split; no independent test set or AR0234 camera footage. Color accuracy is not separately measured."}
    (ROOT / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best.predict(source=str(ROOT / "datasets/balloon/images/val"), device=0,
                 conf=0.25, save=True, project=str(ROOT / "runs"), name="predictions")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
