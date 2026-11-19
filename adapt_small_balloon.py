"""Adapt small pale-green balloons; keep original validation and deployment gated."""
import json
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.yolo'))


def main():
    import cv2
    import numpy as np
    import yaml
    from ultralytics import YOLO
    dataset=ROOT/'datasets/balloon_small_green'
    if dataset.exists():
        raise RuntimeError('Dataset already exists; do not overwrite an earlier experiment')
    shutil.copytree(ROOT/'datasets/balloon_camera_varied',dataset)
    frame=cv2.imread(str(ROOT/'diagnostics/balloon_880_raw.jpg'))
    if frame is None or frame.shape[:2]!=(480,640):
        raise ValueError('Expected manually inspected 640x480 frame')
    # All five actual balloons, including the user-confirmed missed pale green.
    boxes=np.array([[295,239,308,254],[306,240,322,255],[210,255,227,271],
                    [222,331,259,365],[354,262,373,279]],dtype=float)
    def save(name,image,labels):
        height,width=image.shape[:2]
        cv2.imwrite(str(dataset/'images/train'/f'{name}.jpg'),image)
        lines=[]
        for x1,y1,x2,y2 in labels:
            lines.append(f'0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
        (dataset/'labels/train'/f'{name}.txt').write_text('\n'.join(lines)+'\n')
    for i in range(16):save(f'green_full_{i:02}',frame,boxes)
    rng=np.random.default_rng(20261119)
    for i in range(32):
        size=int(rng.integers(100,251))
        x=int(np.clip(301-size/2+rng.integers(-20,21),0,640-size))
        y=int(np.clip(246-size/2+rng.integers(-20,21),0,480-size))
        labels=[]
        for a,b,c,d in boxes:
            left,top,right,bottom=max(a,x),max(b,y),min(c,x+size),min(d,y+size)
            if right>left and bottom>top:
                labels.append([left-x,top-y,right-x,bottom-y])
        save(f'green_context_{i:02}',frame[y:y+size,x:x+size],labels)
    config={'path':str(dataset).replace('\\','/'),'train':'images/train','val':'images/val','names':{0:'balloon'}}
    (dataset/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    (dataset/'small_green_provenance.json').write_text(json.dumps({
        'source_frame':'diagnostics/balloon_880_raw.jpg','manually_reviewed_boxes':boxes.tolist(),
        'green_confirmed_by_user':True,'independent_new_scenes':1,
        'augmentation':'16 full-frame repetitions + 32 contextual crops; not independent scenes',
        'validation':'unchanged original 13 images; no independent green-balloon test set'},indent=2),encoding='utf-8')
    backup=ROOT/'models/balloon_before_small_green.pt'
    if not backup.exists():shutil.copy2(ROOT/'models/balloon.pt',backup)
    original=YOLO(str(backup))
    baseline=original.val(data=str(dataset/'data.yaml'),imgsz=640,batch=8,device=0,workers=0,plots=False,verbose=False)
    model=YOLO(str(backup))
    model.train(data=str(dataset/'data.yaml'),epochs=24,imgsz=640,batch=8,device=0,workers=0,
                project=str(ROOT/'runs'),name='small_green_adaptation',exist_ok=False,
                optimizer='AdamW',lr0=.0003,lrf=.1,warmup_epochs=1,patience=8,freeze=5,
                hsv_h=.04,hsv_s=.3,hsv_v=.25,degrees=3,translate=.05,scale=.25,
                fliplr=.5,mosaic=0,close_mosaic=0,seed=42,deterministic=True,plots=False)
    def iou(a,b):
        area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area+1e-9)
    report={'baseline_map50':float(baseline.box.map50),'baseline_map':float(baseline.box.map),
            'independent_green_test':False,'candidates':[],'deployed':False}
    for path in dict.fromkeys((model.trainer.best,model.trainer.last)):
        candidate=YOLO(str(path))
        validation=candidate.val(data=str(dataset/'data.yaml'),imgsz=640,batch=8,device=0,workers=0,plots=False,verbose=False)
        result=candidate.predict(frame,conf=.05,imgsz=640,device=0,verbose=False)[0]
        detections=[{'box':b.xyxy[0].tolist(),'confidence':float(b.conf[0])} for b in result.boxes]
        confidences=[max((d['confidence'] for d in detections if iou(d['box'],box)>.4),default=0) for box in boxes]
        row={'path':str(path),'map50':float(validation.box.map50),'map':float(validation.box.map),
             'local_balloon_confidences':confidences,'detections':detections}
        row['passes']=(min(confidences)>=.35 and row['map50']>=report['baseline_map50']-.05
                       and row['map']>=report['baseline_map']-.05)
        report['candidates'].append(row)
    passing=[r for r in report['candidates'] if r['passes']]
    if passing:
        chosen=max(passing,key=lambda r:r['local_balloon_confidences'][0])
        # Prepare only: production weights switch separately with the camera stopped.
        report['recommended_weights']=chosen['path']
    (ROOT/'diagnostics/small_green_training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ADAPTATION_REPORT',json.dumps(report),flush=True)


if __name__=='__main__':main()
