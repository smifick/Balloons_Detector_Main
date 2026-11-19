"use strict";

// Display geometry only: no changes to camera, controller or targeting settings.
class VideoViewport {
  constructor(width = 640, height = 480) {
    this.width = width; this.height = height; this.reset();
  }
  reset() {
    this.zoom = 1; this.cx = this.width / 2; this.cy = this.height / 2;
  }
  resize(width, height) {
    if (width === this.width && height === this.height) return;
    this.cx *= width / this.width; this.cy *= height / this.height;
    this.width = width; this.height = height; this.clamp();
  }
  clamp() {
    const hx = this.width / this.zoom / 2, hy = this.height / this.zoom / 2;
    this.cx = Math.max(hx, Math.min(this.width - hx, this.cx));
    this.cy = Math.max(hy, Math.min(this.height - hy, this.cy));
  }
  crop() {
    const width = this.width / this.zoom, height = this.height / this.zoom;
    return {x: this.cx - width / 2, y: this.cy - height / 2, width, height};
  }
  imagePoint(x, y) {
    const crop = this.crop(); return {x: crop.x + x / this.zoom, y: crop.y + y / this.zoom};
  }
  project(x, y) {
    const crop = this.crop(); return [(x - crop.x) * this.zoom, (y - crop.y) * this.zoom];
  }
  zoomAt(zoom, x = this.width / 2, y = this.height / 2) {
    const point = this.imagePoint(x, y);
    this.zoom = Math.max(1, Math.min(8, zoom));
    this.cx = point.x + (this.width / 2 - x) / this.zoom;
    this.cy = point.y + (this.height / 2 - y) / this.zoom;
    this.clamp();
  }
  pan(dx, dy) {
    this.cx -= dx / this.zoom; this.cy -= dy / this.zoom; this.clamp();
  }
  canvasPoint(clientX, clientY, rect) {
    const scale = Math.min(rect.width / this.width, rect.height / this.height);
    if (!(scale > 0)) return null;
    const x = (clientX - rect.left - (rect.width - this.width * scale) / 2) / scale;
    const y = (clientY - rect.top - (rect.height - this.height * scale) / 2) / scale;
    return x < 0 || y < 0 || x > this.width || y > this.height ? null : {x, y};
  }
}

if (typeof module !== "undefined" && module.exports) module.exports = VideoViewport;
else window.VideoViewport = VideoViewport;
