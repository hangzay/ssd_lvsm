const MANIFEST_PATH = "assets/demo/showcase/manifest.json";

const fallbackDemo = {
  title: "Semantic-spatial generation showcase",
  description:
    "Sparse posed images initialize semantic tokens, while target Plucker rays define the spatial query.",
  inputViews: [
    { label: "Input 1", src: "", camera: [-0.82, 0.28, 0.2] },
    { label: "Input 2", src: "", camera: [-0.22, 0.18, 0.05] },
    { label: "Input 3", src: "", camera: [0.42, 0.26, 0.34] },
    { label: "Input 4", src: "", camera: [0.88, 0.34, 0.7] },
  ],
  targetView: {
    label: "Target view",
    src: "",
    camera: [0.16, 0.48, 0.88],
  },
  video: {
    src: "",
    poster: "",
  },
  comparison: {
    src: "",
  },
  trajectory: [
    [-0.82, 0.28, 0.2],
    [-0.56, 0.32, 0.34],
    [-0.22, 0.36, 0.48],
    [0.16, 0.48, 0.88],
    [0.52, 0.4, 0.72],
    [0.88, 0.34, 0.7],
  ],
};

const title = document.getElementById("demoTitle");
const viewCount = document.getElementById("demoViewCount");
const frameCount = document.getElementById("demoFrameCount");
const inputGrid = document.getElementById("inputViewGrid");
const targetImage = document.getElementById("targetImage");
const targetPlaceholder = document.getElementById("targetPlaceholder");
const video = document.getElementById("demoVideo");
const videoPlaceholder = document.getElementById("demoVideoPlaceholder");
const caption = document.getElementById("demoCaption");
const comparisonImage = document.getElementById("comparisonImage");
const comparisonPlaceholder = document.getElementById("comparisonPlaceholder");
const cameraMap = document.getElementById("cameraMap");
const cameraMapPlaceholder = document.getElementById("cameraMapPlaceholder");
const trackSlider = document.getElementById("trackSlider");
const trackPosition = document.getElementById("trackPosition");
const trackBack = document.getElementById("trackBack");
const trackForward = document.getElementById("trackForward");

let demo = fallbackDemo;
let currentTrackIndex = 0;

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
    return {
      ...fallbackDemo,
      ...manifest,
      inputViews: manifest.inputViews || fallbackDemo.inputViews,
      targetView: manifest.targetView || fallbackDemo.targetView,
      video: manifest.video || fallbackDemo.video,
      comparison: manifest.comparison || fallbackDemo.comparison,
      trajectory: manifest.trajectory || fallbackDemo.trajectory,
    };
  } catch {
    return fallbackDemo;
  }
}

function pointToSvg(point) {
  const [x, y, z] = point;
  const sx = 320 + x * 215;
  const sy = 300 - z * 165 - y * 72;
  return [sx, sy];
}

function setMediaPlaceholder(element, placeholder, label) {
  element.removeAttribute("src");
  element.closest(".media-frame").classList.remove("has-media");
  placeholder.textContent = label;
}

async function setImage(element, placeholder, src, label) {
  setMediaPlaceholder(element, placeholder, label);
  if (src && (await assetExists(src))) {
    element.src = src;
    element.closest(".media-frame").classList.add("has-media");
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

function svgEl(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function drawCamera(group, point, className, label) {
  const [x, y] = pointToSvg(point);
  const camera = svgEl("g", { class: className, "data-label": label });
  const body = svgEl("polygon", {
    points: `${x},${y - 10} ${x - 13},${y + 10} ${x + 13},${y + 10}`,
  });
  const rayLeft = svgEl("line", { x1: x, y1: y, x2: x - 24, y2: y + 42 });
  const rayRight = svgEl("line", { x1: x, y1: y, x2: x + 24, y2: y + 42 });
  const text = svgEl("text", { x: x + 15, y: y - 12 });
  text.textContent = label;
  camera.append(body, rayLeft, rayRight, text);
  group.appendChild(camera);
}

function renderCameraMap() {
  cameraMap.replaceChildren();
  cameraMapPlaceholder.hidden = true;

  const grid = svgEl("g", { class: "map-grid-lines" });
  for (let i = 0; i <= 6; i += 1) {
    const x = 80 + i * 80;
    grid.appendChild(svgEl("line", { x1: x, y1: 58, x2: x, y2: 360 }));
  }
  for (let i = 0; i <= 4; i += 1) {
    const y = 90 + i * 58;
    grid.appendChild(svgEl("line", { x1: 76, y1: y, x2: 564, y2: y }));
  }
  cameraMap.appendChild(grid);

  const trajectoryPoints = demo.trajectory.map(pointToSvg);
  const path = svgEl("polyline", {
    class: "trajectory-line",
    points: trajectoryPoints.map(([x, y]) => `${x},${y}`).join(" "),
  });
  cameraMap.appendChild(path);

  const fan = svgEl("g", { id: "rayFan", class: "ray-fan" });
  cameraMap.appendChild(fan);

  const inputGroup = svgEl("g", { id: "inputCameras" });
  demo.inputViews.forEach((view, index) => {
    drawCamera(inputGroup, view.camera, "input-camera", String(index + 1));
  });
  cameraMap.appendChild(inputGroup);

  const targetGroup = svgEl("g", { id: "currentCamera", class: "current-camera" });
  targetGroup.appendChild(svgEl("circle", { r: 10 }));
  targetGroup.appendChild(svgEl("circle", { r: 22 }));
  targetGroup.appendChild(svgEl("text", { x: 16, y: -16 }));
  cameraMap.appendChild(targetGroup);
  updateTrackPosition(0);
}

function updateRayFan(currentPoint) {
  const fan = document.getElementById("rayFan");
  fan.replaceChildren();
  const [cx, cy] = pointToSvg(currentPoint);
  demo.inputViews.forEach((view) => {
    const [ix, iy] = pointToSvg(view.camera);
    fan.appendChild(svgEl("line", { x1: cx, y1: cy, x2: ix, y2: iy }));
  });
}

function updateTrackPosition(index) {
  const maxIndex = Math.max(demo.trajectory.length - 1, 0);
  currentTrackIndex = Math.max(0, Math.min(index, maxIndex));
  const point = demo.trajectory[currentTrackIndex] || demo.targetView.camera;
  const [x, y] = pointToSvg(point);
  const current = document.getElementById("currentCamera");
  current.setAttribute("transform", `translate(${x} ${y})`);
  current.querySelector("text").textContent = "target";
  updateRayFan(point);
  trackSlider.value = String(currentTrackIndex);
  trackPosition.textContent = `${currentTrackIndex + 1}/${maxIndex + 1}`;
}

function setActiveInput(index) {
  for (const tile of inputGrid.querySelectorAll(".input-view-tile")) {
    tile.classList.toggle("active", tile.dataset.index === String(index));
  }
  for (const camera of cameraMap.querySelectorAll(".input-camera")) {
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
  frameCount.textContent = String(demo.trajectory.length);

  trackSlider.max = String(Math.max(demo.trajectory.length - 1, 0));
  trackSlider.value = "0";
  trackSlider.addEventListener("input", () => updateTrackPosition(Number(trackSlider.value)));
  trackBack.addEventListener("click", () => updateTrackPosition(currentTrackIndex - 1));
  trackForward.addEventListener("click", () => updateTrackPosition(currentTrackIndex + 1));
  video.addEventListener("timeupdate", syncTrackFromVideo);

  await Promise.all([
    renderInputViews(),
    setImage(targetImage, targetPlaceholder, demo.targetView.src, demo.targetView.label || "Target view"),
    setVideo(demo.video.src, demo.video.poster),
    setImage(comparisonImage, comparisonPlaceholder, demo.comparison.src, "Target / prediction strip"),
  ]);

  renderCameraMap();
  setActiveInput(0);
}

loadDemo();
