/* MiniLLM chat client: text only, talks to the local stdlib server. */
(() => {
  const $ = (id) => document.getElementById(id);
  const messagesEl = $("messages");
  const inputEl = $("input");
  const sendEl = $("send");
  const bannerEl = $("banner");
  const hintEl = $("hint");

  const state = { messages: [], busy: false, info: null };

  const params = () => ({
    temperature: Number($("temperature").value),
    top_k: Number($("top-k").value),
    max_new_tokens: Number($("max-tokens").value),
    context_limit: Number($("context").value),
    system: $("system").value,
  });

  function bindOutput(id, outputId, format) {
    const input = $(id);
    const output = $(outputId);
    const update = () => { output.textContent = format(input.value); };
    input.addEventListener("input", update);
    update();
  }

  async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  function addMessage(role, content, stats) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "你" : "M";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = content;
    if (stats) {
      const line = document.createElement("div");
      line.className = "stats";
      line.innerHTML = [
        `TTFT ${stats.ttft_ms.toFixed(0)} ms`,
        `预填充 ${stats.prefill_tok_s.toFixed(0)} tok/s`,
        `解码 ${stats.tok_s.toFixed(1)} tok/s`,
        `${stats.tokens} tokens`,
        `总耗时 ${stats.total_ms.toFixed(0)} ms`,
        stats.dropped_turns ? `丢弃 ${stats.dropped_turns} 轮旧对话` : "",
      ].filter(Boolean).map((text) => `<span>${text}</span>`).join("");
      bubble.appendChild(line);
    }
    wrap.append(avatar, bubble);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function setBusy(busy) {
    state.busy = busy;
    sendEl.disabled = busy;
    sendEl.textContent = busy ? "生成中…" : "发送";
    inputEl.disabled = busy;
  }

  function renderInfo(info) {
    state.info = info;
    $("model-card").innerHTML = `
      <dt>名称</dt><dd>${info.name}</dd>
      <dt>参数量</dt><dd>${(info.params / 1e6).toFixed(2)} M</dd>
      <dt>量化</dt><dd>${info.quantized ? `INT${info.quantized}` : "FP32"}</dd>
      <dt>权重文件</dt><dd>${info.checkpoint_human}</dd>
      <dt>层数 / 隐藏维</dt><dd>${info.layers} / ${info.hidden_size}</dd>
      <dt>词表</dt><dd>${info.vocab_size.toLocaleString()}</dd>
      <dt>上下文</dt><dd>${info.max_context}</dd>
      <dt>设备</dt><dd>${info.device}</dd>`;
    $("system").value = info.default_system || "";
    $("context").max = String(info.max_context);
    $("context").value = String(Math.min(Number($("context").value), info.max_context));
    $("context").dispatchEvent(new Event("input"));
    const badges = $("badges");
    badges.innerHTML = `
      <span class="badge ok">${(info.params / 1e6).toFixed(1)}M 参数</span>
      <span class="badge">${info.quantized ? `INT${info.quantized}` : "FP32"} · ${info.checkpoint_human}</span>
      <span class="badge muted">${info.device}</span>`;
    if (!info.trained) {
      badges.insertAdjacentHTML("beforeend", '<span class="badge warn">未训练</span>');
      bannerEl.textContent = "该检查点尚未训练（随机初始化），回复没有语义，只用于验证推理链路。";
      bannerEl.classList.remove("hidden");
    }
  }

  async function send(text) {
    if (!text.trim() || state.busy) return;
    setBusy(true);
    bannerEl.classList.toggle("hidden", Boolean(state.info && state.info.trained));
    addMessage("user", text);
    state.messages.push({ role: "user", content: text });
    $("empty")?.remove();
    const pending = addMessage("assistant", "生成中…");
    pending.classList.add("typing");
    try {
      const result = await fetchJSON("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, ...params() }),
      });
      pending.classList.remove("typing");
      pending.textContent = result.reply || "(空回复)";
      const stats = { ...result };
      const line = document.createElement("div");
      line.className = "stats";
      line.innerHTML = [
        `TTFT ${stats.ttft_ms.toFixed(0)} ms`,
        `预填充 ${stats.prefill_tok_s.toFixed(0)} tok/s`,
        `解码 ${stats.tok_s.toFixed(1)} tok/s`,
        `${stats.tokens} tokens`,
        `总耗时 ${stats.total_ms.toFixed(0)} ms`,
        stats.dropped_turns ? `丢弃 ${stats.dropped_turns} 轮旧对话` : "",
      ].filter(Boolean).map((item) => `<span>${item}</span>`).join("");
      if (result.dropped_turns) {
        $("hint").textContent = `上下文上限 ${result.context_limit} tokens，已丢弃 ${result.dropped_turns} 轮最早对话。`;
      }
      pending.appendChild(line);
      state.messages.push({ role: "assistant", content: result.reply, stats: result });
      $("session-stats").textContent = `${result.history_turns} 轮对话 · 最近一次 ${stats.tokens} tokens`;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } catch (error) {
      pending.textContent = `出错了：${error.message}`;
      pending.classList.add("error");
    } finally {
      setBusy(false);
    }
  }

  function download(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function exportJSON() {
    download("minillm-chat.json",
      JSON.stringify({ model: state.info, messages: state.messages }, null, 2),
      "application/json");
  }

  function exportMarkdown() {
    const lines = [`# MiniLLM 对话记录`, "", `模型：${state.info?.name ?? "unknown"}`, ""];
    for (const message of state.messages) {
      lines.push(`**${message.role === "user" ? "用户" : "助手"}**`, "", message.content, "");
      if (message.stats) {
        lines.push(`> ${message.stats.tokens} tokens · TTFT ${message.stats.ttft_ms.toFixed(0)} ms · ` +
          `解码 ${message.stats.tok_s.toFixed(1)} tok/s`, "");
      }
    }
    download("minillm-chat.md", lines.join("\n"), "text/markdown");
  }

  async function reset() {
    await fetchJSON("/api/reset", { method: "POST", body: "{}" });
    state.messages = [];
    messagesEl.innerHTML = "";
    $("session-stats").textContent = "0 轮对话";
    $("hint").textContent = "仅监听 127.0.0.1 · 无图片输入 · 无联网";
  }

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = inputEl.value;
    inputEl.value = "";
    inputEl.style.height = "auto";
    send(text);
  });
  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
  });
  document.querySelectorAll(".suggestions button").forEach((button) => {
    button.addEventListener("click", () => send(button.dataset.prompt));
  });
  $("clear").addEventListener("click", reset);
  $("export-json").addEventListener("click", exportJSON);
  $("export-md").addEventListener("click", exportMarkdown);

  bindOutput("temperature", "temperature-value", (v) => Number(v).toFixed(2));
  bindOutput("top-k", "top-k-value", (v) => String(v));
  bindOutput("max-tokens", "max-tokens-value", (v) => String(v));
  bindOutput("context", "context-value", (v) => String(v));

  fetchJSON("/api/info").then(renderInfo).catch((error) => {
    bannerEl.textContent = `无法连接本地服务：${error.message}`;
    bannerEl.classList.remove("hidden");
  });
})();
