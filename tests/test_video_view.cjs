const {test} = require('node:test');
const assert = require('node:assert/strict');
const View = require('../control_panel/static/video_view.js');
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);

test('zoom around the cursor preserves the indicated image point',() => {
  const v = new View();
  for (const zoom of [2,4,8,4,2,1]) {
    const before = v.imagePoint(270,220);
    v.zoomAt(zoom,270,220);
    const after = v.imagePoint(270,220);
    close(after.x,before.x); close(after.y,before.y);
    const projected = v.project(after.x,after.y);
    close(projected[0],270); close(projected[1],220);
  }
});
test('letterboxing and zoomed click map back to camera pixels for calibration',() => {
  const v = new View();
  v.zoomAt(4); v.pan(80,-40);
  const rect = {left:10,top:20,width:1000,height:600}; // 100 px pillarbox on each side
  const original = {x:300,y:250};
  const projected = v.project(original.x,original.y);
  const point = v.canvasPoint(110+projected[0]*1.25,20+projected[1]*1.25,rect);
  const restored = v.imagePoint(point.x,point.y);
  close(restored.x,original.x); close(restored.y,original.y);
  assert.equal(v.canvasPoint(50,200,rect),null);
  close((restored.x/640-.5)*100,-3.125);
  close((restored.y/480-.5)*100,2.0833333333333);
});
test('pan and edge zoom stay inside the image and reset restores full frame',() => {
  const v = new View();
  v.zoomAt(100,0,0);
  assert.equal(v.zoom,8);
  for (const [dx,dy] of [[1e5,1e5],[-1e5,-1e5]]) {
    v.pan(dx,dy);
    const r = v.crop();
    assert.ok(r.x>=0 && r.y>=0 && r.x+r.width<=640 && r.y+r.height<=480);
  }
  v.reset(); assert.deepEqual(v.crop(),{x:0,y:0,width:640,height:480});
});
test('resolution changes preserve normalized focus and inverse mapping',() => {
  const v = new View(); v.zoomAt(4,200,200);
  const before = v.crop();
  v.resize(1280,960);
  const after = v.crop();
  close(after.x,before.x*2); close(after.y,before.y*2);
  const point = v.imagePoint(500,400), displayed = v.project(point.x,point.y);
  close(displayed[0],500); close(displayed[1],400);
});
