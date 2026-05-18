const demo = {
  caption: "Novel-view trajectory generated from sparse posed source views.",
  input: "assets/demo/showcase/input.png",
  video: "assets/demo/showcase/rendered_video.mp4",
  comparison: "assets/demo/showcase/gt_vs_pred.png",
  available: false,
};

const inputImage = document.getElementById("demoInput");
const inputPlaceholder = document.getElementById("demoInputPlaceholder");
const video = document.getElementById("demoVideo");
const videoPlaceholder = document.getElementById("demoVideoPlaceholder");
const comparisonImage = document.getElementById("demoComparison");
const comparisonPlaceholder = document.getElementById("demoComparisonPlaceholder");
const caption = document.getElementById("demoCaption");

async function assetExists(path) {
  try {
    const response = await fetch(path, { method: "HEAD", cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

async function setImage(image, placeholder, path, fallback, available) {
  const surface = image.closest(".media-surface");
  image.removeAttribute("src");
  surface.classList.remove("has-media");
  placeholder.textContent = fallback;

  if (available && (await assetExists(path))) {
    image.src = path;
    surface.classList.add("has-media");
  }
}

async function setVideo(path, available) {
  const surface = video.closest(".media-surface");
  video.removeAttribute("src");
  video.load();
  surface.classList.remove("has-media");
  videoPlaceholder.textContent = "Rendered camera trajectory";

  if (available && (await assetExists(path))) {
    video.src = path;
    video.load();
    surface.classList.add("has-media");
  }
}

async function loadDemo() {
  caption.textContent = demo.caption;

  await Promise.all([
    setImage(inputImage, inputPlaceholder, demo.input, "Input views", demo.available),
    setVideo(demo.video, demo.available),
    setImage(comparisonImage, comparisonPlaceholder, demo.comparison, "Target / prediction comparison", demo.available),
  ]);
}

loadDemo();
