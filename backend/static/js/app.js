// Floor Plan AI — single-page editor.
// Uses Konva (loaded as global from CDN) for canvas rendering.

const CLASS_COLORS = {
  wall: { stroke: "#1a1a1a", fill: "rgba(40, 40, 40, 0.6)" },
  window: { stroke: "#3c8cdc", fill: "rgba(60, 140, 220, 0.4)" },
  door: { stroke: "#dc6438", fill: "rgba(220, 100, 60, 0.4)" },
  room: { stroke: "#5fa05f", fill: "rgba(180, 220, 180, 0.35)" },
};

const state = {
  imageId: null,
  imageElement: null,
  imageWidth: 0,
  imageHeight: 0,
  scale: 1,
  stage: null,
  imageLayer: null,
  polyLayer: null,
  polygons: [], // [{ id, category, points: [[x,y]], shape }]
  activeTool: "select",
  selectedId: null,
  drafting: null, // { points, line, dots, category }
};

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
document.querySelectorAll("nav .tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav .tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    document.getElementById(`tab-${tab}`).classList.add("active");
  });
});

// ---------------------------------------------------------------------------
// Image upload + canvas init
// ---------------------------------------------------------------------------
const imageInput = document.getElementById("image-input");
imageInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("image", file);
  setStatus("detect-status", "Загружаем…");
  const resp = await fetch("/api/upload", { method: "POST", body: fd });
  if (!resp.ok) {
    setStatus("detect-status", "Ошибка загрузки");
    return;
  }
  const data = await resp.json();
  loadImageOnCanvas(data);
  setStatus("detect-status", "Готово. Нажмите «Авто-разметка».");
});

async function loadImageOnCanvas(data) {
  state.imageId = data.image_id;
  state.imageWidth = data.width;
  state.imageHeight = data.height;
  state.polygons = [];

  const img = new Image();
  img.onload = () => {
    state.imageElement = img;
    initStage();
    document.getElementById("auto-detect").disabled = false;
    document.getElementById("export-coco").disabled = false;
    document.getElementById("canvas-empty").style.display = "none";
  };
  img.src = data.data_url;
}

function initStage() {
  const container = document.getElementById("canvas-container");
  container.innerHTML = "";
  const wrapper = container.parentElement;
  const padding = 32;
  const maxW = wrapper.clientWidth - padding;
  const maxH = wrapper.clientHeight - padding;
  state.scale = Math.min(maxW / state.imageWidth, maxH / state.imageHeight, 1);
  const w = Math.round(state.imageWidth * state.scale);
  const h = Math.round(state.imageHeight * state.scale);

  state.stage = new Konva.Stage({ container: "canvas-container", width: w, height: h });
  state.imageLayer = new Konva.Layer();
  state.polyLayer = new Konva.Layer();
  state.stage.add(state.imageLayer);
  state.stage.add(state.polyLayer);

  const k = new Konva.Image({
    image: state.imageElement,
    width: w,
    height: h,
  });
  state.imageLayer.add(k);
  state.imageLayer.draw();

  state.stage.on("click tap", onStageClick);
  state.stage.on("dblclick dbltap", finishDraftingPolygon);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finishDraftingPolygon();
    if (e.key === "Escape") cancelDraftingPolygon();
    if (e.key === "Delete" && state.selectedId !== null) deletePolygon(state.selectedId);
  });
}

// ---------------------------------------------------------------------------
// Drawing polygons
// ---------------------------------------------------------------------------
function getMousePoint() {
  const pos = state.stage.getPointerPosition();
  return [pos.x / state.scale, pos.y / state.scale];
}

function onStageClick(e) {
  if (state.activeTool !== "add") return;
  const cls = currentClass();
  const point = getMousePoint();

  if (!state.drafting) {
    state.drafting = { points: [point], category: cls, line: null, dots: [] };
  } else {
    state.drafting.points.push(point);
  }
  redrawDraft();
}

function redrawDraft() {
  if (!state.drafting) return;
  if (state.drafting.line) state.drafting.line.destroy();
  state.drafting.dots.forEach((d) => d.destroy());
  state.drafting.dots = [];
  const flat = state.drafting.points.flatMap(([x, y]) => [x * state.scale, y * state.scale]);
  state.drafting.line = new Konva.Line({
    points: flat,
    stroke: CLASS_COLORS[state.drafting.category].stroke,
    strokeWidth: 2,
    dash: [4, 3],
    listening: false,
  });
  state.polyLayer.add(state.drafting.line);
  state.drafting.points.forEach(([x, y]) => {
    const dot = new Konva.Circle({
      x: x * state.scale,
      y: y * state.scale,
      radius: 3,
      fill: CLASS_COLORS[state.drafting.category].stroke,
      listening: false,
    });
    state.drafting.dots.push(dot);
    state.polyLayer.add(dot);
  });
  state.polyLayer.draw();
}

function finishDraftingPolygon() {
  if (!state.drafting || state.drafting.points.length < 3) return;
  const poly = {
    id: cryptoRandomId(),
    category: state.drafting.category,
    points: state.drafting.points,
  };
  state.polygons.push(poly);
  state.drafting.line && state.drafting.line.destroy();
  state.drafting.dots.forEach((d) => d.destroy());
  state.drafting = null;
  drawPolygon(poly);
  state.polyLayer.draw();
}

function cancelDraftingPolygon() {
  if (!state.drafting) return;
  state.drafting.line && state.drafting.line.destroy();
  state.drafting.dots.forEach((d) => d.destroy());
  state.drafting = null;
  state.polyLayer.draw();
}

function drawPolygon(poly) {
  const colors = CLASS_COLORS[poly.category] || CLASS_COLORS.wall;
  const flat = poly.points.flatMap(([x, y]) => [x * state.scale, y * state.scale]);
  const shape = new Konva.Line({
    points: flat,
    stroke: colors.stroke,
    fill: colors.fill,
    strokeWidth: 2,
    closed: true,
    name: poly.id,
  });
  shape.on("click tap", () => {
    if (state.activeTool === "delete") {
      deletePolygon(poly.id);
      return;
    }
    selectPolygon(poly.id);
  });
  state.polyLayer.add(shape);
  poly.shape = shape;

  // Vertex handles for editing.
  poly.handles = poly.points.map((pt, idx) => {
    const handle = new Konva.Circle({
      x: pt[0] * state.scale,
      y: pt[1] * state.scale,
      radius: 5,
      fill: "white",
      stroke: colors.stroke,
      strokeWidth: 1.5,
      draggable: true,
      visible: false,
    });
    handle.on("dragmove", () => {
      poly.points[idx] = [handle.x() / state.scale, handle.y() / state.scale];
      shape.points(poly.points.flatMap(([x, y]) => [x * state.scale, y * state.scale]));
      state.polyLayer.batchDraw();
    });
    state.polyLayer.add(handle);
    return handle;
  });
}

function selectPolygon(id) {
  state.selectedId = id;
  state.polygons.forEach((p) => {
    const isSelected = p.id === id;
    p.shape.strokeWidth(isSelected ? 3 : 2);
    p.shape.dash(isSelected ? [6, 4] : []);
    p.handles.forEach((h) => h.visible(isSelected));
  });
  state.polyLayer.draw();
}

function deletePolygon(id) {
  const idx = state.polygons.findIndex((p) => p.id === id);
  if (idx < 0) return;
  const p = state.polygons[idx];
  p.shape.destroy();
  p.handles.forEach((h) => h.destroy());
  state.polygons.splice(idx, 1);
  if (state.selectedId === id) state.selectedId = null;
  state.polyLayer.draw();
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------
const tools = [
  ["tool-select", "select"],
  ["tool-add", "add"],
  ["tool-delete", "delete"],
];
tools.forEach(([id, name]) => {
  document.getElementById(id).addEventListener("click", () => {
    cancelDraftingPolygon();
    state.activeTool = name;
    document.querySelectorAll(".tool-buttons .tool").forEach((b) => b.classList.remove("active"));
    document.getElementById(id).classList.add("active");
  });
});

function currentClass() {
  return document.querySelector('input[name="class"]:checked').value;
}

// ---------------------------------------------------------------------------
// Auto-detection
// ---------------------------------------------------------------------------
document.getElementById("auto-detect").addEventListener("click", async () => {
  if (!state.imageId) return;
  setStatus("detect-status", "Распознаём…");
  const resp = await fetch("/api/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: state.imageId }),
  });
  if (!resp.ok) {
    setStatus("detect-status", "Ошибка распознавания");
    return;
  }
  const data = await resp.json();
  setStatus("detect-status", `Найдено ${data.polygons.length} объектов (${data.model}).`);
  document.getElementById("model-badge").textContent = `segmentation: ${data.model}`;
  data.polygons.forEach((p) => {
    const poly = { id: cryptoRandomId(), category: p.category, points: p.points };
    state.polygons.push(poly);
    drawPolygon(poly);
  });
  state.polyLayer.draw();
});

// ---------------------------------------------------------------------------
// Clear / export / import COCO
// ---------------------------------------------------------------------------
document.getElementById("clear-annotations").addEventListener("click", () => {
  state.polygons.forEach((p) => {
    p.shape.destroy();
    p.handles.forEach((h) => h.destroy());
  });
  state.polygons = [];
  state.selectedId = null;
  state.polyLayer.draw();
});

document.getElementById("export-coco").addEventListener("click", async () => {
  if (!state.imageId) return;
  const payload = {
    images: [
      {
        file_name: `${state.imageId}.png`,
        width: state.imageWidth,
        height: state.imageHeight,
        polygons: state.polygons.map((p) => ({ category: p.category, points: p.points })),
      },
    ],
  };
  const resp = await fetch("/api/export/coco", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    alert("Ошибка экспорта COCO");
    return;
  }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "annotations.coco.json";
  a.click();
  URL.revokeObjectURL(url);
});

document.getElementById("import-coco").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file || !state.imageId) return;
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch("/api/import/coco", { method: "POST", body: fd });
  if (!resp.ok) {
    alert("Не удалось распарсить COCO");
    return;
  }
  const data = await resp.json();
  if (!data.images.length) return;
  // Replace current annotations with the first image's polygons.
  state.polygons.forEach((p) => {
    p.shape.destroy();
    p.handles.forEach((h) => h.destroy());
  });
  state.polygons = [];
  data.images[0].polygons.forEach((p) => {
    const poly = { id: cryptoRandomId(), category: p.category, points: p.points };
    state.polygons.push(poly);
    drawPolygon(poly);
  });
  state.polyLayer.draw();
});

// ---------------------------------------------------------------------------
// Generation tab
// ---------------------------------------------------------------------------
const genImage = document.getElementById("gen-image");
const genEmpty = document.getElementById("gen-empty");
let lastGenerated = null;

document.getElementById("gen-run").addEventListener("click", async () => {
  setStatus("gen-status", "Генерируем…");
  const body = {
    width: Number(document.getElementById("gen-width").value),
    height: Number(document.getElementById("gen-height").value),
    num_rooms: Number(document.getElementById("gen-rooms").value),
    boundary_shape: document.getElementById("gen-shape").value,
  };
  const area = Number(document.getElementById("gen-area").value);
  if (area > 0) body.area_m2 = area;
  const seed = document.getElementById("gen-seed").value;
  if (seed !== "") body.seed = Number(seed);

  const resp = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    setStatus("gen-status", "Ошибка генерации");
    return;
  }
  const data = await resp.json();
  genImage.src = data.data_url;
  genImage.classList.add("visible");
  genEmpty.style.display = "none";
  setStatus("gen-status", `Готово (${data.model}).`);
  document.getElementById("model-badge").textContent = `generation: ${data.model}`;
  lastGenerated = data;
  document.getElementById("gen-to-editor").disabled = false;
});

document.getElementById("gen-to-editor").addEventListener("click", async () => {
  if (!lastGenerated) return;
  // Upload data URL as a new editor image.
  const resp = await fetch("/api/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_url: lastGenerated.data_url }),
  });
  if (!resp.ok) return;
  const data = await resp.json();
  loadImageOnCanvas(data);
  // Pre-load polygons from generation as editable annotations.
  lastGenerated.polygons.forEach((p) => {
    const poly = { id: cryptoRandomId(), category: p.category, points: p.points };
    state.polygons.push(poly);
    drawPolygon(poly);
  });
  state.polyLayer.draw();
  // Switch tab.
  document.querySelector('nav .tab[data-tab="annotate"]').click();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function setStatus(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg;
}

function cryptoRandomId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return Math.random().toString(36).slice(2);
}

// Display backend health on startup.
fetch("/health")
  .then((r) => r.json())
  .then((h) => {
    document.getElementById("model-badge").textContent =
      `segmentation: ${h.segmenter} · generation: ${h.generator}`;
  })
  .catch(() => {});
