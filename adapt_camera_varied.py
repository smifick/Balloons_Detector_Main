"""Train with real local annotations plus varied-position cutout augmentation.

Synthetic images are augmentation, not additional independently collected scenes.
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
    import numpy as np
    from ultralytics import YOLO

    target = ROOT / "datasets/balloon_camera_varied"
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            destination = target / kind / split
            destination.mkdir(parents=True, exist_ok=True)
            for source in (ROOT / "datasets/balloon" / kind / split).iterdir():
                if source.is_file():
                    shutil.copy2(source, destination / source.name)
    old = cv2.imread(str(ROOT / "diagnostics/cap_frame.jpg"))
    current = cv2.imread(str(ROOT / "diagnostics/current_scene.jpg"))
    scenes = [(old, [(269,290,304,325),(205,252,224,271)]),
              (current, [(168,184,182,199),(150,244,167,261)])]
    def label_lines(boxes):
        return "".join(f"0 {(a+c)/1280:.8f} {(b+d)/960:.8f} {(c-a)/640:.8f} {(d-b)/480:.8f}\n" for a,b,c,d in boxes)
    def save(name, image, boxes):
        cv2.imwrite(str(target / "images/train" / (name + ".jpg")), image)
        (target / "labels/train" / (name + ".txt")).write_text(label_lines(boxes))
    for index, (frame, boxes) in enumerate(scenes):
        if frame is None or frame.shape[:2] != (480,640):
            raise ValueError("Expected manually inspected 640x480 local scenes")
        for i in range(8):
            save(f"local_{index}_{i:02d}",frame,boxes)
    # Build a room background without its real balloons, then vary the extracted
    # balloon's position, scale and color. The laser cap stays unlabelled background.
    backgrounds = []
    for frame, boxes in scenes:
        mask = np.zeros(frame.shape[:2], np.uint8)
        for a,b,c,d in boxes:
            cv2.rectangle(mask,(a-3,b-3),(c+3,d+3),255,-1)
        backgrounds.append(cv2.inpaint(frame,mask,3,cv2.INPAINT_TELEA))
    patch = old[290:325,269:304].copy()
    alpha = np.zeros((35,35),np.uint8)
    cv2.ellipse(alpha,(17,17),(16,17),0,0,360,255,-1)
    alpha = cv2.GaussianBlur(alpha,(3,3),0.7).astype(np.float32)/255
    rng = np.random.default_rng(42)
    preview = None
    for i in range(96):
        image = backgrounds[i % len(backgrounds)].copy()
        boxes = []
        for j in range(int(rng.integers(1,4))):
            for attempt in range(100):
                size = int(rng.integers(14,82))
                w = max(10,int(size*rng.uniform(.8,1.1))); h = size
                x = int(rng.integers(8,632-w)); y = int(rng.integers(110,470-h))
                if all(x+w<a-4 or x>c+4 or y+h<b-4 or y>d+4 for a,b,c,d in boxes):
                    break
            else:
                continue
            colored = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
            colored[:,:,0] = (colored[:,:,0].astype(np.int16)+int(rng.integers(-70,71)))%180
            colored[:,:,1] = np.clip(colored[:,:,1].astype(np.float32)*rng.uniform(.7,1.2),0,255).astype(np.uint8)
            colored[:,:,2] = np.clip(colored[:,:,2].astype(np.float32)*rng.uniform(.7,1.2),0,255).astype(np.uint8)
            colored = cv2.resize(cv2.cvtColor(colored,cv2.COLOR_HSV2BGR),(w,h))
            opacity = cv2.resize(alpha,(w,h))[:,:,None]
            image[y:y+h,x:x+w] = (colored*opacity+image[y:y+h,x:x+w]*(1-opacity)).astype(np.uint8)
            boxes.append((x,y,x+w,y+h))
        save(f"synthetic_{i:03d}",image,boxes)
        if i == 0:
            preview = image.copy()
            for a,b,c,d in boxes:
                cv2.rectangle(preview,(a,b),(c,d),(0,255,0),1)
    cv2.imwrite(str(ROOT / "diagnostics/varied_training_example.jpg"),preview)
    (target / "data.yaml").write_text("path: "+json.dumps(target.as_posix(),ensure_ascii=False)+
        "\ntrain: images/train\nval: images/val\nnames:\n  0: balloon\n",encoding="utf-8")
    (target / "provenance.json").write_text(json.dumps({
        "original_train_images":61,"independent_local_frames":2,"local_frame_oversampling":8,
        "synthetic_train_images":96,"validation":"Original 13 images; synthetic images are training augmentation only.",
        "local_labels":[boxes for frame,boxes in scenes]},indent=2))
    model=YOLO(str(ROOT / "runs/balloon_camera_two_targets/weights/best.pt"))
    model.train(data=str(target / "data.yaml"),epochs=40,imgsz=640,batch=8,device=0,workers=0,
                amp=False,project=str(ROOT / "runs"),name="balloon_camera_varied",exist_ok=False,
                patience=0,optimizer="AdamW",lr0=.0005,lrf=.1,warmup_epochs=2,
                hsv_h=.25,hsv_s=.4,hsv_v=.3,degrees=5,translate=.1,scale=.3,
                fliplr=.5,mosaic=.3,close_mosaic=10,seed=42,deterministic=True,plots=True)
    print("Trained weights:",model.trainer.best,flush=True)


if __name__ == "__main__":
    main()
