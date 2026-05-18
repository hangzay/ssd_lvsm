const MANIFEST_PATH = "assets/demo/showcase/manifest.json";

const fallbackDemo = {
  title: "Semantic-spatial generation showcase",
  description:
    "Two posed inputs initialize semantic tokens, while three target Plucker-ray queries define the output views.",
  inputViews: [
    { label: "Input 1", src: "", camera: [-0.72, 0.22, 0.18] },
    { label: "Input 2", src: "", camera: [0.72, 0.24, 0.18] },
  ],
  outputViews: [
    { label: "Output 1", src: "", camera: [-0.38, 0.3, 0.62] },
    { label: "Output 2", src: "", camera: [0.02, 0.37, 0.78] },
    { label: "Output 3", src: "", camera: [0.44, 0.31, 0.62] },
  ],
  video: {
    src: "",
    poster: "",
  },
  trajectory: [
    [-0.38, 0.3, 0.62],
    [0.02, 0.37, 0.78],
    [0.44, 0.31, 0.62],
  ],
};

const title = document.getElementById("demoTitle");
const viewCount = document.getElementById("demoViewCount");
const frameCount = document.getElementById("demoFrameCount");
const inputGrid = document.getElementById("inputViewGrid");
const outputGrid = document.getElementById("outputViewGrid");
const video = document.getElementById("demoVideo");
const videoPlaceholder = document.getElementById("demoVideoPlaceholder");
const caption = document.getElementById("demoCaption");
const cameraMap = document.getElementById("cameraMap");
const cameraMapPlaceholder = document.getElementById("cameraMapPlaceholder");
const trackSlider = document.getElementById("trackSlider");
const trackPosition = document.getElementById("trackPosition");
const trackBack = document.getElementById("trackBack");
const trackForward = document.getElementById("trackForward");

let demo = fallbackDemo;
let currentTrackIndex = 0;
let activeInputIndex = 0;
let spatialInteractionReady = false;

const spatialView = {
  yaw: -0.72,
  pitch: -0.56,
  zoom: 1.18,
  isDragging: false,
  lastX: 0,
  lastY: 0,
};

async function assetExists(path) {
  try {
    const response = await fetch(path, { method: "HEAD", cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

async function loadManifest() {
  try {
    const response = await fetch(MANIFEST_PATH, { cache: "no-store" });
    if (!response.ok) {
      return fallbackDemo;
    }
    const manifest = await response.json();
    const outputViews =
      manifest.outputViews ||
      manifest.targetViews ||
      (manifest.targetView ? [manifest.targetView] : fallbackDemo.outputViews);
    const trajectory = manifest.trajectory || outputViews.map((view) => view.camera).filter(Boolean);
    return {
      ...fallbackDemo,
      ...manifest,
      inputViews: manifest.inputViews || fallbackDemo.inputViews,
      outputViews,
      video: manifest.video || fallbackDemo.video,
      trajectory: trajectory.length > 0 ? trajectory : fallbackDemo.trajectory,
    };
  } catch {
    return fallbackDemo;
  }
}

async function setVideo(src, poster) {
  video.removeAttribute("src");
  video.removeAttribute("poster");
  video.load();
  video.closest(".media-frame").classList.remove("has-media");
  videoPlaceholder.textContent = "Rendered trajectory";

  if (poster && (await assetExists(poster))) {
    video.poster = poster;
  }
  if (src && (await assetExists(src))) {
    video.src = src;
    video.load();
    video.closest(".media-frame").classList.add("has-media");
  }
}

async function renderInputViews() {
  inputGrid.replaceChildren();
  const tiles = await Promise.all(
    demo.inputViews.map(async (view, index) => {
      const tile = document.createElement("button");
      tile.className = "input-view-tile";
      tile.type = "button";
      tile.dataset.index = String(index);
      tile.setAttribute("aria-label", `Highlight ${view.label}`);

      const frame = document.createElement("span");
      frame.className = "input-thumb-frame";
      const label = document.createElement("span");
      label.className = "input-thumb-label";
      label.textContent = view.label || `Input ${index + 1}`;

      if (view.src && (await assetExists(view.src))) {
        const img = document.createElement("img");
        img.src = view.src;
        img.alt = view.label || `Input view ${index + 1}`;
        frame.classList.add("has-media");
        frame.appendChild(img);
      } else {
        frame.textContent = view.label || `Input ${index + 1}`;
      }

      tile.append(frame, label);
      tile.addEventListener("mouseenter", () => setActiveInput(index));
      tile.addEventListener("focus", () => setActiveInput(index));
      tile.addEventListener("click", () => setActiveInput(index));
      return tile;
    })
  );

  for (const tile of tiles) {
    inputGrid.appendChild(tile);
  }
}

async function renderOutputViews() {
  outputGrid.replaceChildren();
  const tiles = await Promise.all(
    demo.outputViews.map(async (view, index) => {
      const tile = document.createElement("button");
      tile.className = "output-view-tile";
      tile.type = "button";
      tile.dataset.index = String(index);
      tile.setAttribute("aria-label", `Select ${view.label}`);

      const frame = document.createElement("span");
      frame.className = "output-thumb-frame";
      const label = document.createElement("span");
      label.className = "output-thumb-label";
      label.textContent = view.label || `Output ${index + 1}`;

      if (view.src && (await assetExists(view.src))) {
        const img = document.createElement("img");
        img.src = view.src;
        img.alt = view.label || `Output view ${index + 1}`;
        frame.classList.add("has-media");
        frame.appendChild(img);
      } else {
        frame.textContent = view.label || `Output ${index + 1}`;
      }

      tile.append(frame, label);
      tile.addEventListener("mouseenter", () => updateTrackPosition(index));
      tile.addEventListener("focus", () => updateTrackPosition(index));
      tile.addEventListener("click", () => updateTrackPosition(index));
      return tile;
    })
  );

  for (const tile of tiles) {
    outputGrid.appendChild(tile);
  }
}

function svgEl(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function vecAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vecSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vecScale(a, scale) {
  return [a[0] * scale, a[1] * scale, a[2] * scale];
}

function vecLength(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

function vecNormalize(a, fallback = [0, 0, 1]) {
  const length = vecLength(a);
  if (length < 0.0001) {
    return fallback;
  }
  return [a[0] / length, a[1] / length, a[2] / length];
}

function vecCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function rotateWorld(point) {
  const centered = [point[0], point[1] * 1.28 - 0.1, point[2] - 0.48];
  const cosY = Math.cos(spatialView.yaw);
  const sinY = Math.sin(spatialView.yaw);
  const yawedX = centered[0] * cosY - centered[2] * sinY;
  const yawedZ = centered[0] * sinY + centered[2] * cosY;

  const cosP = Math.cos(spatialView.pitch);
  const sinP = Math.sin(spatialView.pitch);
  const pitchedY = centered[1] * cosP - yawedZ * sinP;
  const pitchedZ = centered[1] * sinP + yawedZ * cosP;

  return [yawedX, pitchedY, pitchedZ];
}

function projectPoint(point) {
  const rotated = rotateWorld(point);
  const distance = 3.05;
  const depth = distance + rotated[2];
  const perspective = (640 * spatialView.zoom) / depth;
  return {
    x: 320 + rotated[0] * perspective,
    y: 236 - rotated[1] * perspective,
    depth: rotated[2],
    scale: perspective / 260,
  };
}

function pointsToAttribute(points) {
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
}

function draw3dLine(group, from, to, className, extra = {}) {
  const a = projectPoint(from);
  const b = projectPoint(to);
  group.appendChild(svgEl("line", { class: className, x1: a.x, y1: a.y, x2: b.x, y2: b.y, ...extra }));
}

function drawSceneVolume() {
  const volume = svgEl("g", { class: "scene-volume" });
  const floorCorners = [
    projectPoint([-1.08, -0.06, -0.08]),
    projectPoint([1.08, -0.06, -0.08]),
    projectPoint([1.08, -0.06, 1.08]),
    projectPoint([-1.08, -0.06, 1.08]),
  ];
  volume.appendChild(svgEl("polygon", { class: "map-floor", points: pointsToAttribute(floorCorners) }));

  const backCorners = [
    projectPoint([-1.08, -0.06, 1.08]),
    projectPoint([1.08, -0.06, 1.08]),
    projectPoint([1.08, 0.64, 1.08]),
    projectPoint([-1.08, 0.64, 1.08]),
  ];
  volume.appendChild(svgEl("polygon", { class: "map-backdrop", points: pointsToAttribute(backCorners) }));

  const grid = svgEl("g", { class: "map-grid-lines" });
  for (let i = 0; i <= 8; i += 1) {
    const x = -1.08 + (i / 8) * 2.16;
    draw3dLine(grid, [x, -0.055, -0.08], [x, -0.055, 1.08], "grid-line");
  }
  for (let i = 0; i <= 6; i += 1) {
    const z = -0.08 + (i / 6) * 1.16;
    draw3dLine(grid, [-1.08, -0.055, z], [1.08, -0.055, z], "grid-line");
  }
  for (let i = 1; i <= 3; i += 1) {
    const y = -0.06 + i * 0.2;
    draw3dLine(grid, [-1.08, y, 1.08], [1.08, y, 1.08], "wall-grid-line");
  }
  volume.appendChild(grid);

  const axes = svgEl("g", { class: "scene-axes" });
  draw3dLine(axes, [-1.08, -0.04, -0.08], [1.08, -0.04, -0.08], "axis-line axis-x");
  draw3dLine(axes, [-1.08, -0.04, -0.08], [-1.08, 0.64, -0.08], "axis-line axis-y");
  draw3dLine(axes, [-1.08, -0.04, -0.08], [-1.08, -0.04, 1.08], "axis-line axis-z");
  volume.appendChild(axes);

  return volume;
}

function createFrustum(view, index, type) {
  const center = view.camera || fallbackDemo.outputViews[0].camera;
  const focus = [0, 0.04, 0.48];
  const forward = vecNormalize(vecSub(focus, center));
  let right = vecNormalize(vecCross(forward, [0, 1, 0]), [1, 0, 0]);
  let up = vecNormalize(vecCross(right, forward), [0, 1, 0]);
  right = vecScale(right, type === "current" ? 0.12 : 0.095);
  up = vecScale(up, type === "current" ? 0.082 : 0.066);

  const near = vecAdd(center, vecScale(forward, type === "current" ? 0.18 : 0.15));
  const corners = [
    vecAdd(vecAdd(near, right), up),
    vecAdd(vecSub(near, right), up),
    vecSub(vecSub(near, right), up),
    vecSub(vecAdd(near, right), up),
  ];
  const projectedCenter = projectPoint(center);
  const projectedCorners = corners.map(projectPoint);
  const label = type === "input" ? `in ${index + 1}` : `out ${index + 1}`;
  const classes = [
    "camera-frustum",
    type === "input" ? "input-camera" : "output-camera",
    type === "current" ? "current-camera active" : "",
    type === "input" && index === activeInputIndex ? "active" : "",
    type === "output" && index === currentTrackIndex ? "active" : "",
  ].filter(Boolean).join(" ");
  const camera = svgEl("g", { class: classes, "data-label": String(index + 1) });
  camera.appendChild(svgEl("ellipse", {
    class: "camera-shadow",
    cx: projectedCenter.x,
    cy: projectedCenter.y + 18 * projectedCenter.scale,
    rx: 18 * projectedCenter.scale,
    ry: 5 * projectedCenter.scale,
  }));
  camera.appendChild(svgEl("polygon", {
    class: "frustum-face",
    points: pointsToAttribute(projectedCorners),
  }));
  projectedCorners.forEach((corner) => {
    camera.appendChild(svgEl("line", {
      class: "frustum-edge",
      x1: projectedCenter.x,
      y1: projectedCenter.y,
      x2: corner.x,
      y2: corner.y,
    }));
  });
  camera.appendChild(svgEl("circle", {
    class: "camera-center",
    cx: projectedCenter.x,
    cy: projectedCenter.y,
    r: clamp(5.5 * projectedCenter.scale, 3.8, 8.5),
  }));
  const text = svgEl("text", {
    x: projectedCenter.x + 16 * projectedCenter.scale,
    y: projectedCenter.y - 14 * projectedCenter.scale,
  });
  text.textContent = label;
  camera.appendChild(text);

  return {
    element: camera,
    depth: projectedCenter.depth + projectedCorners.reduce((sum, point) => sum + point.depth, 0) / projectedCorners.length,
  };
}

function renderTrajectoryLayer() {
  const layer = svgEl("g", { class: "trajectory-layer" });
  const projected = demo.trajectory.map(projectPoint);
  if (projected.length > 1) {
    layer.appendChild(svgEl("polyline", {
      class: "trajectory-shadow",
      points: pointsToAttribute(projected.map((point) => ({ ...point, y: point.y + 4 }))),
    }));
    layer.appendChild(svgEl("polyline", {
      class: "trajectory-line",
      points: pointsToAttribute(projected),
    }));
  }
  projected.forEach((point, index) => {
    layer.appendChild(svgEl("circle", {
      class: `trajectory-knot${index === currentTrackIndex ? " active" : ""}`,
      cx: point.x,
      cy: point.y,
      r: index === currentTrackIndex ? 5 : 3.5,
    }));
  });
  return layer;
}

function renderRayFanLayer(currentPoint) {
  const fan = svgEl("g", { class: "ray-fan" });
  demo.inputViews.forEach((view) => {
    draw3dLine(fan, currentPoint, view.camera, "ray-line");
  });
  return fan;
}

function getCurrentTrajectoryPoint() {
  return (
    demo.trajectory[currentTrackIndex] ||
    demo.outputViews[currentTrackIndex]?.camera ||
    fallbackDemo.outputViews[0].camera
  );
}

function renderCameraMap() {
  cameraMap.replaceChildren();
  cameraMapPlaceholder.hidden = true;

  const currentPoint = getCurrentTrajectoryPoint();
  cameraMap.appendChild(drawSceneVolume());
  cameraMap.appendChild(renderRayFanLayer(currentPoint));
  cameraMap.appendChild(renderTrajectoryLayer());

  const cameraItems = [];
  demo.inputViews.forEach((view, index) => {
    cameraItems.push(createFrustum(view, index, "input"));
  });
  demo.outputViews.forEach((view, index) => {
    cameraItems.push(createFrustum(view, index, index === currentTrackIndex ? "current" : "output"));
  });
  cameraItems
    .sort((a, b) => a.depth - b.depth)
    .forEach((item) => cameraMap.appendChild(item.element));
}

function setupSpatialMapInteractions() {
  if (spatialInteractionReady) {
    return;
  }
  spatialInteractionReady = true;

  cameraMap.addEventListener("pointerdown", (event) => {
    spatialView.isDragging = true;
    spatialView.lastX = event.clientX;
    spatialView.lastY = event.clientY;
    cameraMap.setPointerCapture(event.pointerId);
  });

  cameraMap.addEventListener("pointermove", (event) => {
    if (!spatialView.isDragging) {
      return;
    }
    const dx = event.clientX - spatialView.lastX;
    const dy = event.clientY - spatialView.lastY;
    spatialView.lastX = event.clientX;
    spatialView.lastY = event.clientY;
    spatialView.yaw += dx * 0.008;
    spatialView.pitch = clamp(spatialView.pitch + dy * 0.006, -1.15, -0.18);
    renderCameraMap();
  });

  cameraMap.addEventListener("pointerup", (event) => {
    spatialView.isDragging = false;
    cameraMap.releasePointerCapture(event.pointerId);
  });

  cameraMap.addEventListener("pointercancel", () => {
    spatialView.isDragging = false;
  });

  cameraMap.addEventListener("wheel", (event) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.08 : 0.08;
    spatialView.zoom = clamp(spatialView.zoom + delta, 0.72, 2.1);
    renderCameraMap();
  }, { passive: false });

  cameraMap.addEventListener("dblclick", () => {
    spatialView.yaw = -0.72;
    spatialView.pitch = -0.56;
    spatialView.zoom = 1.18;
    renderCameraMap();
  });

  cameraMap.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      spatialView.yaw += event.key === "ArrowLeft" ? -0.08 : 0.08;
    } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      spatialView.pitch = clamp(spatialView.pitch + (event.key === "ArrowUp" ? -0.06 : 0.06), -1.15, -0.18);
    } else if (event.key === "+" || event.key === "=" || event.key === "-") {
      spatialView.zoom = clamp(spatialView.zoom + (event.key === "-" ? -0.08 : 0.08), 0.72, 2.1);
    } else {
      return;
    }
    event.preventDefault();
    renderCameraMap();
  });
}

function updateTrackPosition(index) {
  const maxIndex = Math.max(demo.trajectory.length - 1, 0);
  currentTrackIndex = Math.max(0, Math.min(index, maxIndex));
  trackSlider.value = String(currentTrackIndex);
  trackPosition.textContent = `${currentTrackIndex + 1}/${maxIndex + 1}`;
  renderCameraMap();
  setActiveOutput(currentTrackIndex);
}

function setActiveInput(index) {
  activeInputIndex = Math.max(0, Math.min(index, Math.max(demo.inputViews.length - 1, 0)));
  for (const tile of inputGrid.querySelectorAll(".input-view-tile")) {
    tile.classList.toggle("active", tile.dataset.index === String(activeInputIndex));
  }
  for (const camera of cameraMap.querySelectorAll(".input-camera")) {
    camera.classList.toggle("active", camera.dataset.label === String(activeInputIndex + 1));
  }
}

function setActiveOutput(index) {
  for (const tile of outputGrid.querySelectorAll(".output-view-tile")) {
    tile.classList.toggle("active", tile.dataset.index === String(index));
  }
  for (const camera of cameraMap.querySelectorAll(".output-camera")) {
    camera.classList.toggle("active", camera.dataset.label === String(index + 1));
  }
}

function syncTrackFromVideo() {
  if (!Number.isFinite(video.duration) || video.duration <= 0 || demo.trajectory.length <= 1) {
    return;
  }
  const progress = video.currentTime / video.duration;
  updateTrackPosition(Math.round(progress * (demo.trajectory.length - 1)));
}

async function loadDemo() {
  demo = await loadManifest();
  title.textContent = demo.title || fallbackDemo.title;
  caption.textContent = demo.description || fallbackDemo.description;
  viewCount.textContent = String(demo.inputViews.length);
  frameCount.textContent = String(demo.outputViews.length);

  trackSlider.max = String(Math.max(demo.trajectory.length - 1, 0));
  trackSlider.value = "0";
  trackSlider.addEventListener("input", () => updateTrackPosition(Number(trackSlider.value)));
  trackBack.addEventListener("click", () => updateTrackPosition(currentTrackIndex - 1));
  trackForward.addEventListener("click", () => updateTrackPosition(currentTrackIndex + 1));
  video.addEventListener("timeupdate", syncTrackFromVideo);
  setupSpatialMapInteractions();

  await Promise.all([
    renderInputViews(),
    renderOutputViews(),
    setVideo(demo.video.src, demo.video.poster),
  ]);

  renderCameraMap();
  setActiveInput(0);
  updateTrackPosition(0);
}

loadDemo();
