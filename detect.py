"""Run balloon detection on an image, video, or a USB/UVC camera."""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_CONFIDENCE = 0.35
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def run_camera(model, index, args):
    import cv2
    from detection_filter import filter_result, load_regions

    backends = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    cap = cv2.VideoCapture(index, backends[args.backend])
    writer = None
    title = f"Balloon detector - camera {index} - Q / Esc to exit"
    frames = detections = excluded = 0
    started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise ConnectionError(f"Camera {index} could not be opened ({args.backend}). Try --source 1 or --source 2.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ConnectionError(f"Camera {index} opened but returned no frame ({args.backend}). Try --backend dshow or another camera index.")
        print(f"Camera {index}: {frame.shape[1]}x{frame.shape[0]}, backend={cap.getBackendName()}", flush=True)
        print(f"Display confidence threshold: {args.conf:.2f}", flush=True)
        regions = [] if args.no_exclusions else load_regions(args.exclusions, index, frame.shape)
        print("Exclusion regions: " + ", ".join(region["name"] for region in regions), flush=True)
        if not args.no_show:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if args.save:
            folder = ROOT / "runs/detect"
            folder.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1 <= fps <= 240 else 30
            filename = folder / f"camera_{index}_{time.time_ns()}.mp4"
            writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video file: {filename}")
            print(f"Saving video: {filename}", flush=True)
        while True:
            result = model.predict(source=frame, conf=args.conf, imgsz=640, device=0, verbose=False)[0]
            # Apply the display threshold before plotting or recording any boxes.
            result.update(boxes=result.boxes.data[result.boxes.conf >= args.conf])
            excluded += filter_result(result, regions)
            frames += 1
            detections += len(result.boxes)
            if not args.no_show or writer is not None:
                annotated = result.plot()
                if writer is not None:
                    writer.write(annotated)
                if not args.no_show:
                    cv2.imshow(title, annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        break
            if frames % 30 == 0:
                print(f"Frames={frames}, detections={detections}, excluded={excluded}, average_fps={frames/(time.perf_counter()-started):.1f}", flush=True)
            if args.max_frames and frames >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ConnectionError(f"Camera {index} stopped returning frames after {frames} frames.")
        print(f"Completed: frames={frames}, detections={detections}, excluded={excluded}, average_fps={frames/(time.perf_counter()-started):.1f}", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="UVC camera index or image/video path")
    active_weights = ROOT / "models/balloon.pt"
    parser.add_argument("--weights", default=str(active_weights if active_weights.is_file()
                        else ROOT / "runs/balloon_yolov8n/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=MIN_CONFIDENCE,
                        help="Minimum confidence shown (0.35 to 1)")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"),
                        default="dshow" if os.name == "nt" else "auto",
                        help="OpenCV camera backend (Windows default: dshow)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop a camera check after N frames; 0 means unlimited")
    parser.add_argument("--exclusions", default=str(ROOT / "camera_exclusions.json"), help="Fixed hardware exclusion regions per camera")
    parser.add_argument("--no-exclusions", action="store_true", help="Disable calibrated hardware exclusion regions")
    args = parser.parse_args()
    if not MIN_CONFIDENCE <= args.conf <= 1:
        parser.error("--conf must be at least 0.35 and at most 1")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if not Path(args.weights).is_file():
        parser.error(f"Weights not found: {args.weights}")
    from ultralytics import YOLO
    import cv2
    source = int(args.source) if args.source.isdecimal() else args.source
    try:
        model = YOLO(args.weights)
        if isinstance(source, int):
            run_camera(model, source, args)
        else:
            for result in model.predict(source=source, stream=True,
                     conf=args.conf, imgsz=640, device=0, show=not args.no_show,
                     save=args.save, project=str(ROOT / "runs"), name="detect", verbose=False):
                for box in result.boxes:
                    print({"xyxy": box.xyxy[0].tolist(), "confidence": float(box.conf[0])})
    except (ConnectionError, FileNotFoundError, RuntimeError, ValueError, KeyError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
