const demos = [
  {
    id: "representation",
    label: "Representation",
    title: "Representation analysis",
    description:
      "Add a video showing semantic and spatial branch behavior across views or layers.",
    video: "assets/demo/representation-analysis.mp4",
    poster: "assets/demo/representation-analysis.jpg",
    available: false,
  },
  {
    id: "results",
    label: "NVS Results",
    title: "Novel view synthesis results",
    description:
      "Add a video showing target-view synthesis or side-by-side baseline comparisons.",
    video: "assets/demo/nvs-results.mp4",
    poster: "assets/demo/nvs-results.jpg",
    available: false,
  },
];

const controls = document.getElementById("demoControls");
const title = document.getElementById("demoTitle");
const description = document.getElementById("demoDescription");
const assetHint = document.getElementById("demoAssetHint");
const frame = document.getElementById("demoFrame");
const video = document.getElementById("demoVideo");
const placeholder = document.getElementById("demoPlaceholder");

async function assetExists(path) {
  try {
    const response = await fetch(path, { method: "HEAD", cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

async function selectDemo(demo) {
  title.textContent = demo.title;
  description.textContent = demo.description;
  assetHint.textContent = `Expected video: ${demo.video}`;
  placeholder.innerHTML = `Add <code>${demo.video}</code> to activate this slot.`;

  for (const button of controls.querySelectorAll(".demo-button")) {
    button.classList.toggle("active", button.dataset.demoId === demo.id);
  }

  frame.classList.remove("has-video");
  video.removeAttribute("src");
  video.removeAttribute("poster");
  video.load();

  if (demo.available && (await assetExists(demo.video))) {
    video.src = demo.video;
    if (await assetExists(demo.poster)) {
      video.poster = demo.poster;
    }
    video.load();
    frame.classList.add("has-video");
  }
}

for (const demo of demos) {
  const button = document.createElement("button");
  button.className = "demo-button";
  button.type = "button";
  button.dataset.demoId = demo.id;
  button.textContent = demo.label;
  button.addEventListener("click", () => selectDemo(demo));
  controls.appendChild(button);
}

selectDemo(demos[0]);
