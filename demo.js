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

function pointToSvg(point) {
  const [x, y, z] = point;
  const sx = 320 + x * 185 + z * 86;
  const sy = 332 + x * 34 - z * 104 - y * 118;
  return [sx, sy];
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

function drawPlaneGrid() {
  const plane = svgEl("g", { class: "map-plane" });
  const corners = [
    pointToSvg([-1, 0, 0]),
    pointToSvg([1, 0, 0]),
    pointToSvg([1, 0, 1]),
    pointToSvg([-1, 0, 1]),
  ];
  plane.appendChild(
    svgEl("polygon", {
      class: "map-floor",
      points: corners.map(([x, y]) => `${x},${y}`).join(" "),
    })
  );

  const grid = svgEl("g", { class: "map-grid-lines" });
  for (let i = 0; i <= 6; i += 1) {
    const x = -1 + (i / 6) * 2;
    const [x1, y1] = pointToSvg([x, 0, 0]);
    const [x2, y2] = pointToSvg([x, 0, 1]);
    grid.appendChild(svgEl("line", { x1, y1, x2, y2 }));
  }
  for (let i = 0; i <= 5; i += 1) {
    const z = i / 5;
    const [x1, y1] = pointToSvg([-1, 0, z]);
    const [x2, y2] = pointToSvg([1, 0, z]);
    grid.appendChild(svgEl("line", { x1, y1, x2, y2 }));
  }
  plane.appendChild(grid);
  return plane;
}

function buildCameraGlyph(className, label) {
  const camera = svgEl("g", { class: `camera-glyph ${className}`, "data-label": label });
  const left = "-18,16";
  const right = "18,16";
  const bottom = "0,38";

  camera.appendChild(svgEl("polygon", { class: "pyramid-face pyramid-left", points: `0,-16 ${left} ${bottom}` }));
  camera.appendChild(svgEl("polygon", { class: "pyramid-face pyramid-right", points: `0,-16 ${right} ${bottom}` }));
  camera.appendChild(svgEl("polygon", { class: "pyramid-base", points: `${left} ${right} ${bottom}` }));
  camera.appendChild(svgEl("line", { class: "pyramid-center", x1: 0, y1: -16, x2: 0, y2: 38 }));
  const text = svgEl("text", { x: 22, y: -20 });
  text.textContent = label;
  camera.appendChild(text);
  return camera;
}

function drawCamera(group, point, className, label) {
  const [x, y] = pointToSvg(point);
  const camera = buildCameraGlyph(className, label);
  camera.setAttribute("transform", `translate(${x} ${y})`);
  group.appendChild(camera);
}

function renderCameraMap() {
  cameraMap.replaceChildren();
  cameraMapPlaceholder.hidden = true;

  cameraMap.appendChild(drawPlaneGrid());

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

  const outputGroup = svgEl("g", { id: "outputCameras" });
  demo.outputViews.forEach((view, index) => {
    drawCamera(outputGroup, view.camera, "output-camera", String(index + 1));
  });
  cameraMap.appendChild(outputGroup);

  const currentGroup = buildCameraGlyph("current-camera", "out 1");
  currentGroup.setAttribute("id", "currentCamera");
  cameraMap.appendChild(currentGroup);
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
  const point =
    demo.trajectory[currentTrackIndex] ||
    demo.outputViews[currentTrackIndex]?.camera ||
    fallbackDemo.outputViews[0].camera;
  const [x, y] = pointToSvg(point);
  const current = document.getElementById("currentCamera");
  current.setAttribute("transform", `translate(${x} ${y})`);
  current.querySelector("text").textContent = `out ${currentTrackIndex + 1}`;
  updateRayFan(point);
  trackSlider.value = String(currentTrackIndex);
  trackPosition.textContent = `${currentTrackIndex + 1}/${maxIndex + 1}`;
  setActiveOutput(currentTrackIndex);
}

function setActiveInput(index) {
  for (const tile of inputGrid.querySelectorAll(".input-view-tile")) {
    tile.classList.toggle("active", tile.dataset.index === String(index));
  }
  for (const camera of cameraMap.querySelectorAll(".input-camera")) {
    camera.classList.toggle("active", camera.dataset.label === String(index + 1));
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

  await Promise.all([
    renderInputViews(),
    renderOutputViews(),
    setVideo(demo.video.src, demo.video.poster),
  ]);

  renderCameraMap();
  setActiveInput(0);
  setActiveOutput(0);
}

loadDemo();
