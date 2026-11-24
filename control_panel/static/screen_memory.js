"use strict";

class ScreenBalloonMemory {
  static memoryMs = 4000;
  static forecastMs = 450;

  constructor() {
    this.reset();
  }

  reset(selection = null) {
    this.selection = selection;
    this.reference = null;
    this.pending = null;
    this.history = [];
    this.colorCandidate = null;
  }

  update(selection, tracks, now, describe) {
    if (this.selection !== selection) this.reset(selection);
    if (selection === null) return null;
    if (this.reference && now - this.reference.seen > ScreenBalloonMemory.memoryMs) this.reset(selection);
    if (!this.reference) {
      const track = tracks.find(candidate => candidate.id === selection && candidate.center);
      const appearance = track && describe(track);
      if (appearance) this.remember(track, appearance, now);
      return null;
    }

    const previous = this.reference;
    const radius = Math.min(470, Math.max(40, previous.size * 2) + 0.36 * (now - previous.seen));
    const matches = [];
    for (const track of tracks) {
      if (!track.center) continue;
      const [left, top, right, bottom] = track.box;
      const width = right - left, height = bottom - top;
      const size = Math.sqrt(width * height);
      const sizeChange = Math.abs(Math.log(size / previous.size));
      const shapeChange = Math.abs(Math.log((width / height) / previous.aspect));
      const distance = Math.hypot(track.center[0] - previous.center[0], track.center[1] - previous.center[1]);
      if (sizeChange > 0.72 || shapeChange > 0.48 || distance > radius) continue;
      const appearance = describe(track);
      if (!appearance) continue;
      const colorDistance = appearance.reduce((sum, value, index) => sum + Math.abs(value - previous.appearance[index]), 0) / 2;
      if (colorDistance > 0.34) continue;
      const score = colorDistance + 0.24 * distance / radius + 0.16 * sizeChange + 0.12 * shapeChange;
      matches.push({track, appearance, score});
    }
    matches.sort((a, b) => a.score - b.score);
    const [match, alternative] = matches;
    if (!match || (alternative && alternative.score - match.score < 0.10)) {
      this.pending = null;
      return this.estimate(now);
    }
    const {track, appearance} = match;
    if (track.id !== previous.id || now - previous.seen > 180) {
      if (this.pending?.id !== track.id) this.pending = {id: track.id, stamp: null, count: 0};
      if (track.observed_at !== this.pending.stamp) {
        this.pending.stamp = track.observed_at;
        this.pending.count += 1;
      }
      if (this.pending.count < 3) return this.estimate(now);
    }
    this.pending = null;
    this.remember(track, appearance, now);
    return track.id !== selection ? track : null;
  }

  remember(track, appearance, now) {
    if (track.observed_at === this.reference?.stamp && track.id === this.reference?.id) return;
    this.colorCandidate = null;
    const [left, top, right, bottom] = track.box;
    const width = right - left, height = bottom - top;
    if (track.id !== this.reference?.id || now - this.reference.seen > 300) this.history = [];
    this.history.push({stamp: track.observed_at, center: track.center});
    this.history = this.history.filter(sample => track.observed_at - sample.stamp <= 0.30).slice(-4);
    this.reference = {
      id: track.id, stamp: track.observed_at, seen: now, center: track.center,
      size: Math.sqrt(width * height), aspect: width / height,
      appearance: this.reference?.appearance ?? appearance
    };
  }

  forecast(now) {
    if (!this.reference || this.history.length < 2) return null;
    const elapsed = now - this.reference.seen;
    if (elapsed <= 0 || elapsed > ScreenBalloonMemory.forecastMs) return null;
    const times = this.history.map(sample => sample.stamp - this.reference.stamp);
    const mean = times.reduce((sum, time) => sum + time, 0) / times.length;
    const centered = times.map(time => time - mean);
    const squareSum = centered.reduce((sum, time) => sum + time * time, 0);
    if (squareSum < 0.00005) return null;
    const velocity = this.reference.center.map((_, axis) =>
      centered.reduce((sum, time, index) => sum + time * this.history[index].center[axis], 0) / squareSum
    );
    const travelScale = Math.min(elapsed / 1000, 150 / Math.max(1, Math.hypot(...velocity)));
    return {
      predicted: true,
      center: this.reference.center.map((value, axis) => value + velocity[axis] * travelScale),
      uncertainty: Math.max(7, this.reference.size * 0.30) + elapsed * 0.035
    };
  }

  estimate(now) {
    if (!this.reference) return null;
    return this.forecast(now) ?? {
      lastSeen: true,
      center: this.reference.center,
      uncertainty: Math.max(10, this.reference.size * 0.65)
    };
  }

  followOrange(now, locate) {
    const previous = this.reference;
    if (!previous || now - previous.seen > 1200) return null;
    const orange = previous.appearance[0] + previous.appearance[1] + previous.appearance[2];
    if (orange < 0.5) return null;
    const recent = this.colorCandidate && now - this.colorCandidate.time <= 150 ? this.colorCandidate : null;
    const horizon = recent ? Math.min(100, now - recent.time) : 0;
    const anchor = recent ? recent.center.map((value, axis) => value + recent.velocity[axis] * horizon)
                         : this.forecast(now)?.center ?? previous.center;
    const radius = recent ? Math.max(25, previous.size * 1.5) :
      Math.min(235, Math.max(36, previous.size * 1.7) + (now - previous.seen) * 0.18);
    const center = locate(anchor, radius, previous.size);
    if (!center) {
      this.colorCandidate = null;
      return null;
    }
    const elapsed = recent ? now - recent.time : 0;
    const continuous = recent && elapsed > 0 &&
      Math.hypot(center[0] - anchor[0], center[1] - anchor[1]) <= Math.max(20, previous.size);
    this.colorCandidate = {
      center, time: now, count: continuous ? recent.count + 1 : 1,
      velocity: continuous ? center.map((value, axis) => (value - recent.center[axis]) / elapsed) : [0, 0]
    };
    return this.colorCandidate.count >= 2 ? {center, colorOnly: true} : null;
  }
}

function findOrangeRegion(pixels, width, height, anchor, size) {
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i += 1) {
    const j = i * 4;
    const r = pixels[j], g = pixels[j + 1], b = pixels[j + 2];
    mask[i] = r > 70 && r - b >= 28 && r - g >= 10 && g - b >= 8 && g - b < 1.8 * (r - g);
  }
  const queue = new Int32Array(mask.length);
  const matches = [];
  const radius = Math.max(1, Math.min(width, height) / 2);
  const expectedArea = size * size * 0.60;
  for (let seed = 0; seed < mask.length; seed += 1) {
    if (!mask[seed]) continue;
    queue[0] = seed;
    mask[seed] = 0;
    let count = 1, totalX = 0, totalY = 0;
    let left = width, right = 0, top = height, bottom = 0;
    for (let i = 0; i < count; i += 1) {
      const k = queue[i], x = k % width, y = Math.floor(k / width);
      totalX += x + 0.5; totalY += y + 0.5;
      left = Math.min(left, x); right = Math.max(right, x);
      top = Math.min(top, y); bottom = Math.max(bottom, y);
      const neighbors = [x > 0 ? k - 1 : -1, x + 1 < width ? k + 1 : -1, k - width, k + width];
      for (const n of neighbors) {
        if (!mask[n]) continue;
        mask[n] = 0;
        queue[count++] = n;
      }
    }
    const w = right - left + 1, h = bottom - top + 1;
    const ratio = count / expectedArea;
    if (count < 9 || ratio < 0.22 || ratio > 2.35 || w / h < 0.45 || w / h > 2.2 || count / (w * h) < 0.40) continue;
    const center = [totalX / count, totalY / count];
    const distance = Math.hypot(center[0] - anchor[0], center[1] - anchor[1]);
    if (distance > radius) continue;
    matches.push({center, score: distance / radius + 0.24 * Math.abs(Math.log(ratio))});
  }
  matches.sort((a, b) => a.score - b.score);
  if (!matches.length || (matches[1] && matches[1].score - matches[0].score < 0.15)) return null;
  return matches[0].center;
}

class ScreenDetectionHold {
  constructor() {
    this.entries = new Map();
  }

  clear() {
    this.entries.clear();
  }

  update(tracks, now) {
    const visible = new Set(tracks.map(track => track.id));
    for (const track of tracks) {
      const old = this.entries.get(track.id);
      if (old?.track.observed_at !== track.observed_at) this.entries.set(track.id, {track, seen: now});
    }
    const held = [];
    for (const [id, entry] of this.entries) {
      if (now - entry.seen > 900) {
        this.entries.delete(id);
      } else if (!visible.has(id)) {
        const [left, top, right, bottom] = entry.track.box;
        const replaced = tracks.some(({box: other}) => {
          const overlap = Math.max(0, Math.min(right, other[2]) - Math.max(left, other[0]))
                        * Math.max(0, Math.min(bottom, other[3]) - Math.max(top, other[1]));
          return overlap > (right - left) * (bottom - top) * 0.45;
        });
        if (replaced) this.entries.delete(id);
        else held.push(entry.track);
      }
    }
    return held;
  }
}

function balloonColors(pixels) {
  const bins = Array(15).fill(0);
  const side = Math.sqrt(pixels.length / 4);
  let samples = 0;
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      if (Math.hypot(x + 0.5 - side / 2, y + 0.5 - side / 2) > side * 0.39) continue;
      const index = 4 * (y * side + x);
      const r = pixels[index], g = pixels[index + 1], b = pixels[index + 2];
      const max = Math.max(r, g, b), min = Math.min(r, g, b), chroma = max - min;
      if (chroma < 24 || chroma < max * 0.16) {
        bins[12 + Math.min(2, Math.floor(max / 86))] += 1;
      } else {
        const sector = max === r ? (g - b) / chroma : max === g ? 2 + (b - r) / chroma : 4 + (r - g) / chroma;
        const hue = (sector + 6) % 6;
        const bucket = Math.floor(hue * 2);
        const fraction = hue * 2 - bucket;
        bins[bucket] += 1 - fraction;
        bins[(bucket + 1) % 12] += fraction;
      }
      samples += 1;
    }
  }
  return bins.map(value => value / samples);
}

if (typeof module !== "undefined") module.exports = {ScreenBalloonMemory, ScreenDetectionHold, balloonColors, findOrangeRegion};
