"use strict";

const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="control-token"]').content;
const client = crypto.randomUUID();
const canvas = $("video");
const context = canvas.getContext("2d");
const form = $("settings-form");
const cards = new Map();
let state = null;
let displayedTracks = [];
let displayedFrame = -1;
let imagePending = false;
let lastImage = null;
let lastFrameData = null;
const screenMemory = new ScreenBalloonMemory();
const screenDetectionHold = new ScreenDetectionHold();
let heldDetections = [];
const appearanceCanvas = document.createElement("canvas");
appearanceCanvas.width = appearanceCanvas.height = 24;
const appearanceContext = appearanceCanvas.getContext("2d", {willReadFrequently: true});
const colorCanvas = document.createElement("canvas");
const colorContext = colorCanvas.getContext("2d", {willReadFrequently: true});
let rememberedBalloon = null;
let screenCamera = null;
const videoView = new VideoViewport();
let videoDrag = null;
let suppressVideoClick = false;
let simulated = false;
let portList = [];
let busy = false;
let settingsDirty = false;
let settingsSignature = "";
let eventsSignature = "";
let toastTimer = null;
let calibrationMode = false;

function aimPoint(data) {
  return [data.camera.width * (0.5 + (data.settings.aim_x_percent || 0) / 100),
          data.camera.height * (0.5 + (data.settings.aim_y_percent || 0) / 100)];
}

function toast(message, error = false) {
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.add("hidden"), error ? 6500 : 3500);
}

async function api(path, body = {}) {
  const response = await fetch(`/api/${path}`, {
    method: "POST", headers: {"Content-Type": "application/json", "X-Control-Token": token},
    body: JSON.stringify(body), cache: "no-store"
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Проверьте значения параметров";
    throw new Error(detail);
  }
  if (data.camera) render(data);
  return data;
}

async function action(callback) {
  if (busy) return;
  busy = true;
  if (state) render(state);
  try { await callback(); } catch (error) { toast(error.message, true); }
  finally { busy = false; if (state) render(state); }
}

function dot(element, color) { element.className = `status-dot${color ? " " + color : ""}`; }
function number(value, digits = 0) { return Number.isFinite(value) ? value.toFixed(digits) : "—"; }

function loadSettings(values) {
  for (const [key, value] of Object.entries(values)) {
    const input = form.elements.namedItem(key);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = value;
    else input.value = value;
  }
  $("max-speed-value").textContent = `${Number(values.max_speed).toLocaleString("ru-RU")} °/с`;
  settingsSignature = JSON.stringify(values);
  settingsDirty = false;
}

function render(data) {
  state = data;
  const camera = data.camera;
  const controller = data.controller;
  const tracking = data.tracking;
  const connected = controller.connected;
  const motors = controller.motors_enabled;
  if ((!connected || motors !== true) && heldJogKeys.size) releaseKeyboardJog();
  $("motor-status").textContent = !connected ? "Нет подключения" : motors === true ? "Включены" : motors === false ? "Выключены" : "Проверяем состояние…";
  $("motor-status").classList.toggle("powered", connected && motors === true);
  $("toggle-motors").textContent = busy ? "Подождите…" : motors === true ? "Выключить моторы" : "Включить моторы";
  $("toggle-motors").disabled = !connected || busy || typeof motors !== "boolean";
  $("toggle-motors").classList.toggle("powered", connected && motors === true);
  const live = camera.status === "live";
  if (!live) {
    screenMemory.reset();
    screenDetectionHold.clear();
    heldDetections = [];
    rememberedBalloon = null;
  }
  updateViewControls();
  const isSimulated = connected && controller.info?.simulated;
  $("mobile-stop").classList.toggle("hidden", !connected);
  $("simulation-banner").classList.toggle("hidden", !isSimulated);
  $("camera-label").textContent = live ? "В эфире" : camera.status === "starting" ? "Запускается" : camera.status === "error" ? "Ошибка камеры" : "Выключена";
  dot($("camera-dot"), live ? "green" : camera.status === "error" ? "red" : "");
  $("camera-source").value = String(camera.source);
  $("camera-source").disabled = live || camera.status === "starting" || busy;
  $("camera-toggle").textContent = live || camera.status === "starting" ? "Остановить" : "Запустить";
  $("camera-toggle").disabled = busy;
  $("fps").textContent = live ? number(camera.fps, 1) : "—";
  $("inference").textContent = live ? number(camera.inference_ms) : "—";
  $("resolution").textContent = `${camera.width} × ${camera.height}`;
  $("video-resolution").textContent = `CAM ${String(camera.source).padStart(2, "0")} · ${camera.width} × ${camera.height}`;
  $("video-placeholder").classList.toggle("hidden", live && displayedFrame >= 0);
  $("placeholder-title").textContent = camera.status === "error" ? "Камера недоступна" : camera.status === "off" ? "Камера остановлена" : "Подключаем камеру";
  $("placeholder-text").textContent = camera.error || (camera.status === "off" ? "Нажмите «Запустить» под видеопотоком" : "Загружаем детектор воздушных шариков");
  $("video-hint").textContent = tracking.selected_id ? `Цель #${String(tracking.selected_id).padStart(2,"0")} · ${tracking.active ? "наведение включено" : "готова к наведению"}` : "Кликните по шарику, чтобы выбрать цель";
  $("target-count").textContent = data.tracks.length;
  $("error-x").textContent = number(tracking.error_px.x);
  $("error-y").textContent = number(tracking.error_px.y);
  $("confidence-metric").textContent = data.settings.confidence.toFixed(2);
  $("selected-number").textContent = tracking.selected_id ? String(tracking.selected_id).padStart(2,"0") : "—";
  $("selected-title").textContent = tracking.selected_id ? `Шарик #${String(tracking.selected_id).padStart(2,"0")}` : "Цель не выбрана";
  $("selected-subtitle").textContent = tracking.target ? `Уверенность ${Math.round(tracking.target.confidence * 100)}% · цель видна` : tracking.selected_id ? "Ожидаем выбранную цель" : "Нажмите на шарик в кадре";
  $("tracking-reason").textContent = tracking.reason;
  const limited = tracking.active && tracking.limited_axes?.length > 0;
  dot($("tracking-dot"), tracking.active ? limited || !tracking.target ? "amber" : "green" : "");
  $("tracking-state").classList.toggle("limited", Boolean(limited));
  $("start-tracking").disabled = busy || calibrationMode || (!tracking.active && (!connected || motors !== true || !tracking.target?.center || !live));
  $("calibrate-laser").disabled = busy || !live;
  $("calibrate-laser").textContent = calibrationMode ? "Отменить выбор пятна" : "Указать пятно лазера в видео";
  $("calibration-hint").textContent = calibrationMode ? "Разметка скрыта. Кликните по фактическому пятну лазера в видео. Наведение остановлено. Escape — отмена." : "Прицел — сохранённая точка лазера для расстояния калибровки. Серый крест — центр камеры. Автоматического измерения пятна нет.";
  if (calibrationMode) $("video-hint").textContent = "КАЛИБРОВКА: кликните по пятну лазера · Esc — отмена";
  $("start-tracking").title = connected && motors === false ? "Сначала включите моторы в блоке КОНТРОЛЛЕР" : "";
  $("start-tracking").querySelector("span").textContent = tracking.active ? "Приостановить наведение" : "Начать наведение";
  $("clear-target").disabled = tracking.selected_id === null || busy;
  const badge = $("controller-badge");
  badge.textContent = connected ? isSimulated ? "СИМУЛЯЦИЯ" : "ПОДКЛЮЧЁН" : "НЕ ПОДКЛЮЧЁН";
  badge.className = `pill${connected ? isSimulated ? " sim" : " connected" : ""}`;
  $("controller-info").classList.toggle("hidden", !connected);
  $("board-version").textContent = connected ? `SimpleBGC ${controller.info.board_version}` : "—";
  $("firmware").textContent = connected ? controller.info.firmware : "—";
  $("connect-controller").textContent = busy ? "Подождите…" : connected ? "Отключить контроллер" : "Подключить контроллер ↗";
  $("connect-controller").disabled = busy || (!connected && !simulated && !$("serial-port").value);
  $("mode-real").disabled = connected || busy;
  $("mode-sim").disabled = connected || busy;
  $("serial-port").disabled = connected || busy;
  $("baud").disabled = connected || busy;
  $("connection-fields").classList.toggle("hidden", simulated && !connected || isSimulated);
  $("port-description").textContent = controller.error || (connected ? isSimulated ? "Тестирование без COM-порта" : `${controller.info.port} · Serial API ${controller.info.protocol} · ${controller.info.baud.toLocaleString("ru-RU")} бод` : simulated ? "Тестирование логики без подключения платы" : portList.find(port => port.port === $("serial-port").value)?.description || "Подключите контроллер по USB / UART");
  for (const axis of ["yaw", "pitch"]) {
    $(axis + "-angle").replaceChildren(document.createTextNode(connected ? number(controller.offsets[axis],1) : "—"), Object.assign(document.createElement("small"), {textContent:"°"}));
    $(axis + "-command").textContent = `${number(tracking.command[axis],1)} °/с`;
  }
  document.querySelectorAll("[data-pitch]").forEach(button => button.disabled = !connected || motors !== true || busy);
  $("set-home").disabled = !connected || busy;
  if (!settingsDirty && JSON.stringify(data.settings) !== settingsSignature) loadSettings(data.settings);
  $("save-settings").disabled = !settingsDirty || busy;
  $("footer-status").textContent = live ? `${camera.device} · ${Math.round(camera.fps)} FPS · Pitch / Yaw` : "SimpleBGC Serial API · Управление Pitch / Yaw";
  renderTargets(data.tracks, tracking.selected_id);
  renderEvents(data.events);
}

function renderTargets(tracks, selected) {
  const ids = new Set(tracks.map(track => track.id));
  for (const [id, card] of cards) if (!ids.has(id)) { card.remove(); cards.delete(id); }
  $("empty-targets").classList.toggle("hidden", tracks.length > 0);
  for (const track of tracks) {
    let card = cards.get(track.id);
    if (!card) {
      card = document.createElement("button");
      card.className = "target-card";
      const thumbnail = document.createElement("canvas"); thumbnail.width = thumbnail.height = 56;
      const caption = document.createElement("span"); caption.className = "target-caption";
      caption.append(document.createElement("b"), document.createElement("small"));
      const marker = document.createElement("span"); marker.className = "selection-dot";
      card.append(thumbnail, caption, marker);
      card.addEventListener("click", () => action(() => api("target", {track_id:track.id, client})));
      cards.set(track.id, card); $("targets").append(card);
    }
    card.classList.toggle("selected", selected === track.id);
    card.setAttribute("aria-pressed", String(selected === track.id));
    card.querySelector("b").textContent = `Шарик #${String(track.id).padStart(2,"0")}`;
    const position = track.center ? track.center.map(Math.round).join(", ") : "центр не определён";
    card.querySelector("small").textContent = `${Math.round(track.confidence*100)}% · ${position}`;
  }
}

function renderEvents(events) {
  const signature = JSON.stringify(events);
  if (signature === eventsSignature) return;
  eventsSignature = signature;
  $("event-list").replaceChildren(...events.slice(0,16).map(event => {
    const row = document.createElement("div"); row.className = `event-item ${event.level}`;
    const time = document.createElement("time"); time.textContent = event.time;
    const message = document.createElement("span"); message.textContent = event.message;
    row.append(time, message); return row;
  }));
}

function updateViewControls() {
  const live = state?.camera.status === "live";
  $("zoom-level").textContent = `${videoView.zoom.toLocaleString("ru-RU")}×`;
  $("zoom-in").disabled = !live || videoView.zoom >= 8;
  $("zoom-out").disabled = !live || videoView.zoom <= 1;
  $("zoom-reset").disabled = !live || videoView.zoom <= 1;
  canvas.classList.toggle("zoomed",videoView.zoom > 1);
}

function paintFrame() {
  if (!lastImage || !lastFrameData) return;
  const data = lastFrameData;
  const width = data.camera.width, height = data.camera.height;
  videoView.resize(width,height);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const crop = videoView.crop();
  const rect = canvas.getBoundingClientRect();
  const ui = 1 / (Math.min(rect.width/width,rect.height/height) || 1);
  context.drawImage(lastImage,crop.x,crop.y,crop.width,crop.height,0,0,width,height);
  displayedTracks = data.tracks;
  updateViewControls();
  $("video-placeholder").classList.add("hidden");
  if (calibrationMode) return;
  const selected = data.tracking.selected_id;
  for (const track of data.tracks) {
    if (track.confidence < Math.max(0.35,data.settings.confidence)) continue;
    const [a,b,c,d] = track.box;
    const [x1,y1] = videoView.project(a,b), [x2,y2] = videoView.project(c,d);
    const active = selected === track.id;
    const color = active ? "#c3ef98" : "#d2e4e5";
    context.strokeStyle = color; context.lineWidth = (active ? 2 : 1.3)*ui;
    context.strokeRect(x1,y1,x2-x1,y2-y1);
    const label = `#${String(track.id).padStart(2,"0")}  ${Math.round(track.confidence * 100)}%`;
    context.font = `${10*ui}px Consolas, monospace`;
    const textWidth = context.measureText(label).width;
    const labelY = y1 > 22*ui ? y1 - 20*ui : y2 + 4*ui;
    context.fillStyle = active ? "#c3ef98" : "#112025e8";
    context.fillRect(x1-ui,labelY,textWidth+12*ui,17*ui);
    context.fillStyle = active ? "#14210f" : "#edf5ed";
    context.fillText(label,x1+5*ui,labelY+12*ui);
    if (active && track.center) {
      const [x,y] = videoView.project(...track.center);
      context.strokeStyle = color; context.lineWidth = ui;
      context.beginPath(); context.arc(x,y,4*ui,0,Math.PI*2); context.stroke();
      const [ax,ay] = videoView.project(...aimPoint(data));
      const length = Math.hypot(ax-x,ay-y);
      // Keep the actual spot visible: no filled marker on the balloon or aim.
      if (length > 18*ui) {
        context.setLineDash([4*ui,5*ui]); context.strokeStyle = "#73d9ed99"; context.lineWidth = ui;
        context.beginPath(); context.moveTo(x+(ax-x)*7*ui/length,y+(ay-y)*7*ui/length);
        context.lineTo(ax-(ax-x)*8*ui/length,ay-(ay-y)*8*ui/length);
        context.stroke(); context.setLineDash([]);
      }
    }
    const thumbnail = cards.get(track.id)?.querySelector("canvas");
    if (thumbnail) thumbnail.getContext("2d").drawImage(lastImage,Math.max(0,a-4),Math.max(0,b-4),c-a+8,d-b+8,0,0,56,56);
  }
  context.setLineDash([4*ui,4*ui]);
  context.strokeStyle = "#bfcac9"; context.lineWidth = ui;
  for (const track of heldDetections) {
    if (track.id === selected && rememberedBalloon) continue;
    const [left,top,right,bottom] = track.box;
    const [x1,y1] = videoView.project(left,top);
    const [x2,y2] = videoView.project(right,bottom);
    context.strokeRect(x1,y1,x2-x1,y2-y1);
  }
  context.setLineDash([]);
  if (rememberedBalloon && screenMemory.selection === selected) {
    const [imageX,imageY] = rememberedBalloon.center;
    if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
      const [x,y] = videoView.project(imageX,imageY);
      const predicted = rememberedBalloon.predicted;
      const lastSeen = rememberedBalloon.lastSeen;
      const radius = predicted || lastSeen ? rememberedBalloon.uncertainty * videoView.zoom : 7*ui;
      context.strokeStyle = "#ffcf83"; context.lineWidth = 1.5*ui;
      if (predicted || lastSeen) context.setLineDash([4*ui,4*ui]);
      context.beginPath(); context.arc(x,y,radius,0,Math.PI*2); context.stroke();
      context.setLineDash([]);
      context.fillStyle = "#ffcf83"; context.font = `${11*ui}px sans-serif`;
      const label = predicted ? "Прогноз" : lastSeen ? "Был здесь" :
        rememberedBalloon.colorOnly ? "По цвету" : "Память";
      context.fillText(`${label} #${selected}`,x+radius+4*ui,y+14*ui);
    }
  }
  const [cx,cy] = videoView.project(...aimPoint(data));
  if (data.settings.aim_x_percent || data.settings.aim_y_percent) {
    const [mx,my] = videoView.project(width/2,height/2);
    if (Math.hypot(mx-cx,my-cy)>18*ui) {
      context.strokeStyle = "#a0a9ab"; context.lineWidth = ui;
      context.beginPath(); context.moveTo(mx-5*ui,my); context.lineTo(mx+5*ui,my);
      context.moveTo(mx,my-5*ui); context.lineTo(mx,my+5*ui); context.stroke();
    }
  }
  context.strokeStyle = "#73d9ed"; context.lineWidth = ui;
  context.beginPath(); context.arc(cx,cy,15*ui,0,Math.PI*2);
  context.moveTo(cx-26*ui,cy); context.lineTo(cx-9*ui,cy); context.moveTo(cx+9*ui,cy); context.lineTo(cx+26*ui,cy);
  context.moveTo(cx,cy-26*ui); context.lineTo(cx,cy-9*ui); context.moveTo(cx,cy+9*ui); context.lineTo(cx,cy+26*ui); context.stroke();
}

function drawFrame(data) {
  if (!data.frame || data.frame_number <= displayedFrame || imagePending) return;
  imagePending = true;
  const image = new Image();
  image.onload = () => {
    imagePending = false;
    if (data.frame_number < displayedFrame) return;
    displayedFrame = data.frame_number;
    lastImage = image;
    lastFrameData = data;
    const camera = `${data.camera.source}/${data.camera.width}/${data.camera.height}`;
    if (camera !== screenCamera) {
      screenMemory.reset();
      screenDetectionHold.clear();
    }
    screenCamera = camera;
    const now = performance.now();
    heldDetections = screenDetectionHold.update(data.tracks, now);
    // The result is drawn on the canvas only. It never replaces the server selection.
    rememberedBalloon = screenMemory.update(data.tracking.selected_id, data.tracks, now, track => {
      const [left,top,right,bottom] = track.box;
      if (right <= left || bottom <= top) return null;
      appearanceContext.clearRect(0,0,24,24);
      appearanceContext.drawImage(image,left,top,right-left,bottom-top,0,0,24,24);
      return balloonColors(appearanceContext.getImageData(0,0,24,24).data);
    });
    if ((rememberedBalloon?.predicted || rememberedBalloon?.lastSeen) &&
        !data.tracks.some(track => track.id === screenMemory.selection && track.center)) {
      const colorMark = screenMemory.followOrange(now, ([x,y],radius,size) => {
        const left = Math.max(0, Math.floor(x-radius)), top = Math.max(0, Math.floor(y-radius));
        const right = Math.min(data.camera.width, Math.ceil(x+radius));
        const bottom = Math.min(data.camera.height, Math.ceil(y+radius));
        const width = right-left, height = bottom-top;
        if (width < 8 || height < 8) return null;
        const scale = Math.min(1, 192 / Math.max(width,height));
        const w = Math.round(width*scale), h = Math.round(height*scale);
        if (colorCanvas.width !== w) colorCanvas.width = w;
        if (colorCanvas.height !== h) colorCanvas.height = h;
        colorContext.drawImage(image,left,top,width,height,0,0,w,h);
        const center = findOrangeRegion(colorContext.getImageData(0,0,w,h).data,w,h,
                                        [(x-left)*w/width,(y-top)*h/height],size*scale);
        return center && [left+center[0]*width/w,top+center[1]*height/h];
      });
      if (colorMark) rememberedBalloon = colorMark;
    }
    paintFrame();
  };
  image.onerror = () => { imagePending = false; };
  image.src = "data:image/jpeg;base64," + data.frame;
}

function changeZoom(zoom, point) {
  if (state?.camera.status !== "live") return;
  videoView.zoomAt(zoom,point?.x,point?.y);
  paintFrame();
}
$("zoom-in").addEventListener("click",() => changeZoom(videoView.zoom*2));
$("zoom-out").addEventListener("click",() => changeZoom(videoView.zoom/2));
$("zoom-reset").addEventListener("click",() => { videoView.reset(); paintFrame(); });
canvas.addEventListener("wheel",event => {
  if (state?.camera.status !== "live" || !event.deltaY) return;
  const point = videoView.canvasPoint(event.clientX,event.clientY,canvas.getBoundingClientRect());
  if (!point) return;
  event.preventDefault();
  changeZoom(videoView.zoom*(event.deltaY < 0 ? 2 : 0.5),point);
},{passive:false});
canvas.addEventListener("pointerdown",event => {
  suppressVideoClick = false;
  if (event.button !== 0 || videoView.zoom <= 1) return;
  const point = videoView.canvasPoint(event.clientX,event.clientY,canvas.getBoundingClientRect());
  if (!point) return;
  videoDrag = {id:event.pointerId,startX:event.clientX,startY:event.clientY,last:point,moved:false};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove",event => {
  if (!videoDrag || event.pointerId !== videoDrag.id) return;
  const rect = canvas.getBoundingClientRect();
  // Captured drags may leave the canvas; map their delta without edge clipping.
  const scale = Math.min(rect.width/canvas.width,rect.height/canvas.height);
  const point = {x:(event.clientX-rect.left-(rect.width-canvas.width*scale)/2)/scale,
                 y:(event.clientY-rect.top-(rect.height-canvas.height*scale)/2)/scale};
  if (!videoDrag.moved && Math.hypot(event.clientX-videoDrag.startX,event.clientY-videoDrag.startY)<4) return;
  videoDrag.moved = true;
  canvas.classList.add("dragging");
  videoView.pan(point.x-videoDrag.last.x,point.y-videoDrag.last.y);
  videoDrag.last = point;
  paintFrame();
});
function finishVideoDrag(event) {
  if (!videoDrag || event.pointerId !== videoDrag.id) return;
  suppressVideoClick = videoDrag.moved;
  videoDrag = null;
  canvas.classList.remove("dragging");
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
}
canvas.addEventListener("pointerup",finishVideoDrag);
canvas.addEventListener("pointercancel",finishVideoDrag);

canvas.addEventListener("click", event => {
  if (suppressVideoClick) { suppressVideoClick = false; return; }
  if (!state || busy || state.camera.status !== "live") return;
  const point = videoView.canvasPoint(event.clientX,event.clientY,canvas.getBoundingClientRect());
  if (!point) return;
  const {x,y} = videoView.imagePoint(point.x,point.y);
  if (calibrationMode) {
    const aim_x_percent = Number(((x / canvas.width - 0.5) * 100).toFixed(2));
    const aim_y_percent = Number(((y / canvas.height - 0.5) * 100).toFixed(2));
    if (Math.abs(aim_x_percent) > 40 || Math.abs(aim_y_percent) > 40) {
      toast("Пятно должно быть ближе к центру кадра (в пределах ±40%).",true); return;
    }
    action(async () => {
      await api("calibration",{aim_x_percent,aim_y_percent});
      calibrationMode = false; settingsDirty = false; loadSettings(state.settings);
      toast("Поправка сохранена для рабочего расстояния. Наведение остановлено.");
    });
    return;
  }
  const track = displayedTracks.filter(track => {
    const rect = canvas.getBoundingClientRect();
    const ui = 1/(Math.min(rect.width/canvas.width,rect.height/canvas.height)||1);
    const [a,b,c,d] = track.box; const padding = 8*ui/videoView.zoom;
    return x>=a-padding && x<=c+padding && y>=b-padding && y<=d+padding;
  }).sort((a,b) => Math.hypot(x-a.center[0],y-a.center[1])-Math.hypot(x-b.center[0],y-b.center[1]))[0];
  if (track) action(() => api("target",{track_id:track.id,client}));
  else toast("Нажмите на рамку шарика или выберите его в списке под видео");
});

$("start-tracking").addEventListener("click", () => action(() => state?.tracking.active ? api("stop") : api("start",{client})));
$("stop-tracking").addEventListener("click", () => api("stop").catch(error => toast(error.message,true)));
$("mobile-stop").addEventListener("click", () => api("stop").catch(error => toast(error.message,true)));
$("clear-target").addEventListener("click", () => action(() => api("target",{track_id:null,client})));
$("camera-toggle").addEventListener("click", () => action(async () => {
  if (["live","starting"].includes(state?.camera.status)) await api("camera/stop");
  else { displayedFrame = -1; await api("camera/start",{source:Number($("camera-source").value)}); }
}));
$("fullscreen").addEventListener("click", () => {
  const promise = document.fullscreenElement ? document.exitFullscreen() : $("video-stage").requestFullscreen();
  promise.catch(error => toast(error.message,true));
});

async function refreshPorts() {
  const current = $("serial-port").value;
  try {
    const response = await fetch("/api/ports");
    if (!response.ok) throw new Error("Не удалось прочитать COM-порты");
    portList = await response.json();
    $("serial-port").replaceChildren(...(portList.length ? portList : [{port:"",description:"Порты не найдены"}]).map(port => {
      const option = document.createElement("option"); option.value = port.port; option.textContent = port.port || port.description; return option;
    }));
    if (portList.some(port => port.port === current)) $("serial-port").value = current;
    if (state) render(state);
  } catch (error) { toast(error.message,true); }
}
$("refresh-ports").addEventListener("click",refreshPorts);
$("serial-port").addEventListener("change",() => { if(state) render(state); });
for (const [id,mode] of [["mode-real",false],["mode-sim",true]]) $(id).addEventListener("click",() => {
  simulated = mode;
  $("mode-real").classList.toggle("selected",!mode); $("mode-sim").classList.toggle("selected",mode);
  if (state) render(state);
});
$("connect-controller").addEventListener("click",() => action(() => state?.controller.connected ? api("disconnect") : api("connect",{port:$("serial-port").value,baud:Number($("baud").value),protocol:2,simulated})));
$("toggle-motors").addEventListener("click",() => action(() => api("motors",{enabled:!state.controller.motors_enabled})));
document.querySelectorAll("[data-pitch]").forEach(button => button.addEventListener("click",() => action(() => api("jog",{client,pitch:Number(button.dataset.pitch),yaw:Number(button.dataset.yaw)}))));
$("set-home").addEventListener("click",() => action(() => api("home")));
$("calibrate-laser").addEventListener("click",() => action(async () => {
  if (calibrationMode) { calibrationMode = false; return; }
  if (settingsDirty) { toast("Сначала сохраните параметры или обновите страницу.",true); return; }
  await api("stop"); calibrationMode = true;
  $("video-stage").scrollIntoView({behavior:"smooth",block:"center"});
}));
$("reset-aim").addEventListener("click",() => action(async () => {
  await api("calibration",{aim_x_percent:0,aim_y_percent:0});
  calibrationMode = false; loadSettings(state.settings);
  toast("Прицел возвращён в центр камеры. Наведение остановлено.");
}));

form.addEventListener("input",() => {
  settingsDirty = true; $("save-settings").disabled = busy;
  $("max-speed-value").textContent = `${Number($("max-speed").value).toLocaleString("ru-RU")} °/с`;
});
form.addEventListener("submit",event => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const values = {};
  for (const input of form.elements) if (input.name) values[input.name] = input.type === "checkbox" ? input.checked : Number(input.value);
  action(async () => { await api("settings",values); loadSettings(values); toast("Настройки сохранены. Наведение остановлено."); });
});
$("view-log").addEventListener("click",() => { $("event-panel").open = true; $("event-panel").scrollIntoView({behavior:"smooth",block:"center"}); });
$("view-control").addEventListener("click",() => window.scrollTo({top:0,behavior:"smooth"}));
const jogKeys = new Map([["KeyW", [1, 0]], ["KeyA", [0, -1]], ["KeyS", [-1, 0]], ["KeyD", [0, 1]]]);
const heldJogKeys = new Set();
let jogPending = false;
let jogNeedsStop = false;

async function updateKeyboardJog() {
  if (jogPending) return;
  const direction = jogKeys.get(Array.from(heldJogKeys).at(-1));
  if (!direction && !jogNeedsStop) return;
  jogPending = true;
  jogNeedsStop = Boolean(direction);
  try {
    if (direction) {
      const [pitch, yaw] = direction;
      await api("jog", {client, pitch, yaw});
    } else {
      await api("jog/stop", {client});
    }
  } catch (error) {
    heldJogKeys.clear();
    toast(error.message, true);
  } finally {
    jogPending = false;
    if (!heldJogKeys.size && jogNeedsStop) updateKeyboardJog();
  }
}

function releaseKeyboardJog() {
  heldJogKeys.clear();
  updateKeyboardJog();
}

document.addEventListener("keydown",event => {
  const editing = ["INPUT","SELECT","TEXTAREA"].includes(event.target?.tagName) || event.target?.isContentEditable;
  if (event.code === "Escape" || event.code === "Space" && !editing) {
    releaseKeyboardJog();
    calibrationMode = false;
    event.preventDefault(); api("stop").catch(error => toast(error.message,true));
    return;
  }
  if (editing || event.ctrlKey || event.metaKey || event.altKey || event.repeat || !jogKeys.has(event.code)) return;
  if (busy || calibrationMode || !state?.controller.connected || state.controller.motors_enabled !== true) return;
  event.preventDefault();
  heldJogKeys.add(event.code);
  updateKeyboardJog();
});
document.addEventListener("keyup", event => {
  if (heldJogKeys.delete(event.code)) updateKeyboardJog();
});
document.addEventListener("focusin", event => {
  if (event.target.matches("input, select, textarea") || event.target.isContentEditable) releaseKeyboardJog();
});
window.addEventListener("blur", releaseKeyboardJog);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseKeyboardJog();
});
setInterval(updateKeyboardJog, 80);

const stream = new EventSource(`/api/events?client=${encodeURIComponent(client)}`);
let lastPanelRender = -Infinity;
stream.onopen = () => { dot($("connection-dot"),"green"); $("connection-text").textContent = "Локальная панель онлайн"; $("rail-online").classList.add("online"); };
stream.onmessage = event => {
  try {
    const data = JSON.parse(event.data);
    const now = performance.now();
    if (now - lastPanelRender >= 80 || data.camera.status !== state?.camera.status) {
      lastPanelRender = now;
      render(data);
    }
    drawFrame(data);
  }
  catch (error) { console.error("Не удалось отобразить кадр",error); }
};
stream.onerror = () => {
  releaseKeyboardJog();
  dot($("connection-dot"),"red"); $("connection-text").textContent = "Связь с панелью потеряна";
  $("rail-online").classList.remove("online"); $("start-tracking").disabled = true;
};
setInterval(() => { if (state?.tracking.active) api("heartbeat",{client}).catch(() => {}); },400);
window.addEventListener("pagehide",() => {
  releaseKeyboardJog();
  if (state?.tracking.active) fetch("/api/stop",{method:"POST",headers:{"Content-Type":"application/json","X-Control-Token":token},body:"{}",keepalive:true}).catch(() => {});
  stream.close();
});
refreshPorts();
