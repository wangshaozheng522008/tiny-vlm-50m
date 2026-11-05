"use strict";

const $ = (selector) => document.querySelector(selector);
const modes = { text: "纯文字", image: "纯图片", mixed: "图文混合" };
const modeIcons = { text: "text", image: "image", mixed: "layers" };
const state = { mode: "text", image: null, reference: null, busy: false, preparing: false,
  status: null, history: [], selected: null, examples: [] };
let toastTimer;
let loadingTimer;

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  svg.setAttribute("aria-hidden", "true");
  return svg;
}

function toast(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 2600);
}

function showError(message = "") {
  $("#input-error").textContent = message;
  $("#input-error").hidden = !message;
}

async function requestJSON(url, options = {}) {
  let response;
  try { response = await fetch(url, options); }
  catch { throw new Error("无法连接本地模型服务，请确认服务仍在运行。", { cause: "network" }); }
  let data;
  try { data = await response.json(); }
  catch { throw new Error("服务返回了无法读取的响应，请查看终端日志。"); }
  if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）。`);
  return data;
}

function updateControls() {
  const unavailable = state.busy || state.preparing;
  $("#generate-button").disabled = unavailable || !state.status?.ready;
  $("#generate-label").textContent = state.busy ? "正在生成" : state.preparing ? "准备图片中" : state.status?.ready ? "生成回答" : "等待模型连接";
  $("#generate-arrow").hidden = state.busy;
  $("#generate-spinner").hidden = !state.busy;
  $("#output-card").setAttribute("aria-busy", String(state.busy));
  $("#dropzone").setAttribute("aria-disabled", String(unavailable));
  document.querySelectorAll(".mode-tab, .example-card, .history-item, #prompt, #image-input, #remove-image, #new-test, #clear-history, #max-tokens, #temperature, #top-k, #reset-settings").forEach((element) => { element.disabled = unavailable; });
  $("#export-records").disabled = !state.history.length;
}

function setMode(mode) {
  state.mode = mode;
  state.reference = null;
  document.querySelectorAll(".mode-tab").forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-pressed", String(active));
  });
  $("#upload-area").hidden = mode === "text";
  $("#prompt-area").hidden = mode === "image";
  $("#image-only-hint").hidden = mode !== "image";
  $("#prompt").placeholder = mode === "mixed" ? "针对这张图片，问模型一个问题…" : "写下你的问题，或粘贴一段想测试的文字…";
  $("#mode-description").textContent = mode === "mixed" ? "图片和问题会一起发送给模型。" : "让文字成为这次探索的起点。";
  showError();
}

function updateImagePreview() {
  $("#dropzone").hidden = !!state.image;
  $("#image-preview").hidden = !state.image;
  if (state.image) {
    $("#preview-image").src = state.image.data;
    $("#image-name").textContent = state.image.name;
    $("#image-size").textContent = state.image.size >= 1024 * 1024
      ? `${(state.image.size / 1024 / 1024).toFixed(2)} MB` : `${Math.max(1, Math.round(state.image.size / 1024))} KB`;
  } else {
    $("#preview-image").removeAttribute("src");
    $("#image-input").value = "";
  }
}

async function readImage(file) {
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) throw new Error("图片不能超过 8 MB，请选择一张更小的图片。");
  const extension = file.name.split(".").pop().toLowerCase();
  const mimeTypes = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp", gif: "image/gif" };
  const mime = file.type || mimeTypes[extension];
  if (!["image/png", "image/jpeg", "image/webp", "image/gif"].includes(mime)) throw new Error("请上传 PNG、JPEG、WebP 或 GIF 格式的图片。");
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("图片读取失败，请重新选择。"));
    reader.readAsDataURL(new Blob([file], { type: mime }));
  });
  return { data, name: file.name || "粘贴的图片.png", size: file.size };
}

async function chooseImage(file) {
  if (state.busy || state.preparing || !file) return;
  state.preparing = true;
  updateControls();
  try {
    state.image = await readImage(file);
    state.reference = null;
    updateImagePreview();
    showError();
  } catch (error) { showError(error.message); }
  finally { state.preparing = false; updateControls(); }
}

function resetOutput() {
  $("#output-empty").hidden = false;
  $("#output-loading").hidden = true;
  $("#output-result").hidden = true;
  $("#copy-answer").hidden = true;
  $("#output-card").classList.remove("has-result");
}

function resetTest() {
  if (state.busy || state.preparing) return;
  setMode("text");
  state.image = null;
  state.selected = null;
  $("#prompt").value = "";
  $("#character-count").textContent = "0";
  updateImagePreview();
  resetOutput();
  renderHistory();
  closeMenu();
  $("#prompt").focus();
}

function displayResult(record) {
  $("#output-empty").hidden = true;
  $("#output-loading").hidden = true;
  $("#output-result").hidden = false;
  $("#output-card").classList.add("has-result");
  const empty = !record.result.text.trim();
  $("#answer-text").textContent = empty ? "模型提前结束了生成，没有返回可显示的文字。" : record.result.text;
  $("#answer-text").classList.toggle("empty-answer", empty);
  $("#result-mode").textContent = modes[record.mode];
  $("#copy-answer").hidden = empty;
  $("#stat-time").textContent = (record.result.elapsed_ms / 1000).toFixed(2);
  $("#stat-tokens").textContent = record.result.output_tokens;
  $("#stat-speed").textContent = record.result.tokens_per_second.toFixed(1);
  $("#reference-answer").hidden = !record.reference;
  $("#reference-answer").textContent = record.reference ? `示例参考答案：${record.reference}` : "";
  $("#length-note").hidden = !record.result.limit_reached;
}

function renderHistory() {
  const list = $("#history-list");
  list.replaceChildren();
  $("#history-count").textContent = state.history.length;
  if (!state.history.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.append(icon("clock"));
    const title = document.createElement("p");
    title.textContent = "每一次探索，都有迹可循。";
    const subtitle = document.createElement("span");
    subtitle.textContent = "生成回答后，记录会出现在这里。";
    empty.append(title, subtitle);
    list.append(empty);
  }
  state.history.forEach((record) => {
    const button = document.createElement("button");
    button.className = `history-item${record.id === state.selected ? " selected" : ""}`;
    const top = document.createElement("div");
    top.className = "history-item-top";
    const time = document.createElement("span");
    time.textContent = new Date(record.time).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    const mode = document.createElement("span");
    mode.className = "history-mode";
    mode.textContent = modes[record.mode];
    top.append(time, mode);
    const title = document.createElement("p");
    title.textContent = record.prompt || record.image?.name || "图片测试";
    button.append(top, title);
    button.addEventListener("click", () => {
      if (state.busy || state.preparing) return;
      setMode(record.mode);
      $("#prompt").value = record.prompt;
      $("#character-count").textContent = record.prompt.length;
      state.image = record.image;
      state.reference = record.reference;
      state.selected = record.id;
      updateImagePreview();
      displayResult(record);
      renderHistory();
      closeMenu();
      $("#output-title").scrollIntoView({ behavior: "smooth", block: "center" });
    });
    list.append(button);
  });
  updateControls();
}

async function generate(event) {
  event.preventDefault();
  if (state.busy || state.preparing || !state.status?.ready) return;
  const prompt = state.mode === "image" ? "" : $("#prompt").value;
  if (state.mode !== "image" && !prompt.trim()) {
    showError(state.mode === "mixed" ? "请为这张图片输入一个问题。" : "请先输入要测试的文字。");
    $("#prompt").focus();
    return;
  }
  if (state.mode !== "text" && !state.image) { showError("请先上传一张图片，或点击下方图片示例。"); return; }
  const topK = Number($("#top-k").value);
  if (!Number.isInteger(topK) || topK < 1 || topK > 200) { showError("候选词数量需要是 1 到 200 之间的整数。"); $("#top-k").focus(); return; }
  const request = { mode: state.mode, prompt, image: state.mode === "text" ? null : state.image?.data,
    parameters: { max_new_tokens: Number($("#max-tokens").value), temperature: Number($("#temperature").value), top_k: topK } };
  const record = { id: globalThis.crypto?.randomUUID?.() || `test-${Date.now()}-${Math.random().toString(36).slice(2)}`, time: new Date().toISOString(), mode: state.mode,
    prompt, image: state.mode === "text" ? null : state.image, reference: state.reference };
  state.busy = true;
  showError();
  resetOutput();
  $("#output-empty").hidden = true;
  $("#output-loading").hidden = false;
  $("#output-loading p").textContent = "本地模型正在处理你的输入，请稍候。";
  updateControls();
  const started = Date.now();
  loadingTimer = setInterval(() => { $("#output-loading p").textContent = `本地模型正在处理 · 已用时 ${Math.floor((Date.now() - started) / 1000)} 秒`; }, 1000);
  try {
    record.result = await requestJSON("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    state.history.unshift(record);
    state.history = state.history.slice(0, 12);
    state.selected = record.id;
    displayResult(record);
    renderHistory();
  } catch (error) {
    resetOutput();
    showError(error.message);
    if (error.cause === "network") { state.status = null; displayConnection(false); }
  } finally {
    clearInterval(loadingTimer);
    state.busy = false;
    updateControls();
  }
}

function displayConnection(ready, status = null) {
  const label = $("#connection-status");
  label.replaceChildren();
  const dot = document.createElement("span");
  dot.className = `status-dot${ready ? "" : " offline"}`;
  label.append(dot, document.createTextNode(ready ? "模型已连接" : "连接已断开"));
  $("#sidebar-status").className = `status-dot${ready ? "" : " offline"}`;
  $("#sidebar-model-detail").textContent = ready ? `${status.device.toUpperCase()} · 本地推理` : "请检查本地服务";
  if (ready) {
    $("#checkpoint-name").textContent = status.checkpoint;
    $("#model-parameters").textContent = `${status.parameters_m}M`;
    $("#model-size").textContent = `${status.file_mb}MB`;
    $("#device-name").textContent = status.device;
  }
}

async function checkStatus() {
  try {
    state.status = await requestJSON("/api/status");
    displayConnection(true, state.status);
  } catch {
    state.status = null;
    displayConnection(false);
  }
  updateControls();
}

async function useExample(example) {
  if (state.busy || state.preparing) return;
  state.preparing = true;
  setMode(example.mode);
  state.image = null;
  $("#prompt").value = example.prompt;
  $("#character-count").textContent = example.prompt.length;
  updateImagePreview();
  updateControls();
  try {
    if (example.image_url) {
      const response = await fetch(example.image_url);
      if (!response.ok) throw new Error("示例图片读取失败，请尝试上传自己的图片。");
      const blob = await response.blob();
      state.image = await readImage(new File([blob], "示例图片.png", { type: blob.type }));
      updateImagePreview();
    }
    state.reference = example.reference || null;
    state.selected = null;
    resetOutput();
    renderHistory();
    toast("示例已填入，点击「生成回答」即可测试。");
  } catch (error) { showError(error.message); }
  finally { state.preparing = false; updateControls(); }
}

async function loadExamples() {
  try {
    const data = await requestJSON("/api/examples");
    state.examples = data.examples;
    $("#example-list").replaceChildren();
    state.examples.slice(0, 3).forEach((example) => {
      const button = document.createElement("button");
      button.className = "example-card";
      button.dataset.example = example.id;
      const badge = document.createElement("span");
      badge.className = "example-icon";
      badge.append(icon(modeIcons[example.mode]));
      const title = document.createElement("strong");
      title.textContent = example.title;
      const description = document.createElement("p");
      description.textContent = example.description;
      button.append(badge, title, description, icon("arrow"));
      button.addEventListener("click", () => useExample(example));
      $("#example-list").append(button);
    });
    updateControls();
  } catch { $(".examples-section").hidden = true; }
}

function closeMenu() {
  $("#sidebar").classList.remove("mobile-open");
  $("#menu-button").setAttribute("aria-expanded", "false");
}

function updateSliders() {
  $("#max-tokens-value").textContent = $("#max-tokens").value;
  $("#temperature-value").textContent = Number($("#temperature").value).toFixed(1);
}

document.querySelectorAll(".mode-tab").forEach((button) => button.addEventListener("click", () => { if (!state.busy && !state.preparing) setMode(button.dataset.mode); }));
$("#generate-form").addEventListener("submit", generate);
$("#prompt").addEventListener("input", () => { $("#character-count").textContent = $("#prompt").value.length; state.reference = null; showError(); });
$("#image-input").addEventListener("change", (event) => chooseImage(event.target.files[0]));
$("#remove-image").addEventListener("click", () => { state.image = null; state.reference = null; updateImagePreview(); });
$("#dropzone").addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); if (!state.busy && !state.preparing) $("#image-input").click(); } });
$("#dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("#dropzone").classList.add("drag-over"); });
$("#dropzone").addEventListener("dragleave", () => $("#dropzone").classList.remove("drag-over"));
document.addEventListener("dragover", (event) => { if ([...event.dataTransfer.types].includes("Files")) event.preventDefault(); });
document.addEventListener("drop", (event) => {
  if (!event.dataTransfer.files.length) return;
  event.preventDefault();
  $("#dropzone").classList.remove("drag-over");
  if (state.busy || state.preparing) return;
  if (state.mode === "text") setMode($("#prompt").value.trim() ? "mixed" : "image");
  chooseImage(event.dataTransfer.files[0]);
});
document.addEventListener("paste", (event) => {
  const item = [...(event.clipboardData?.items || [])].find((entry) => entry.type.startsWith("image/"));
  if (!item || state.busy || state.preparing || $("#help-dialog").open) return;
  event.preventDefault();
  if (state.mode === "text") setMode($("#prompt").value.trim() ? "mixed" : "image");
  chooseImage(item.getAsFile());
});
$("#new-test").addEventListener("click", resetTest);
$("#clear-history").addEventListener("click", () => { state.history = []; resetTest(); renderHistory(); toast("本次测试记录已清空。"); });
$("#max-tokens").addEventListener("input", updateSliders);
$("#temperature").addEventListener("input", updateSliders);
$("#reset-settings").addEventListener("click", () => { $("#max-tokens").value = 32; $("#temperature").value = 0; $("#top-k").value = 50; updateSliders(); toast("已恢复默认生成参数。"); });
$("#copy-answer").addEventListener("click", async () => {
  const answer = state.history.find((record) => record.id === state.selected)?.result.text;
  if (!answer) return;
  try { await navigator.clipboard.writeText(answer); toast("回答已复制。"); }
  catch { toast("复制失败，可以直接选中回答文字复制。"); }
});
$("#export-records").addEventListener("click", () => {
  const records = state.history.map(({ image, ...record }) => ({ ...record, image: image ? { name: image.name, size: image.size } : null }));
  const data = { model: state.status?.checkpoint, exported_at: new Date().toISOString(), records };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `tiny-vlm-tests-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("已导出测试记录（包含回答与参数，不包含图片内容）。");
});
$("#help-button").addEventListener("click", () => $("#help-dialog").showModal());
$("#close-help").addEventListener("click", () => $("#help-dialog").close());
$("#help-done").addEventListener("click", () => { $("#help-dialog").close(); $("#prompt").focus(); });
$("#help-dialog").addEventListener("click", (event) => { if (event.target === $("#help-dialog")) { const bounds = event.target.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.target.close(); } });
$("#menu-button").addEventListener("click", () => { const open = $("#sidebar").classList.toggle("mobile-open"); $("#menu-button").setAttribute("aria-expanded", String(open)); });
document.addEventListener("click", (event) => { if (!$("#sidebar").contains(event.target) && !$("#menu-button").contains(event.target)) closeMenu(); });
document.addEventListener("keydown", (event) => {
  if ($("#help-dialog").open) return;
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); $("#generate-form").requestSubmit(); }
  else if (event.key.toLowerCase() === "n" && !event.ctrlKey && !event.metaKey && !event.altKey && !["INPUT", "TEXTAREA"].includes(event.target.tagName)) resetTest();
});

Promise.all([checkStatus(), loadExamples()]);
setInterval(checkStatus, 15000);
