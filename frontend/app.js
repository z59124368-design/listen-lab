(() => {
  "use strict";

  const state = {
    projects: [],
    currentProject: null,
    spokenSentences: [],
    activeSentenceId: null,
    playMode: "continuous", // "continuous" | "single" | "loop"
    soloOverride: null,     // { id, start, end, mode: "once" | "loop" }
    autoScroll: true,
    userScrolling: false,
    blindMode: false,
    editingSentenceId: null,
    searchQuery: "",
    loopGapMs: 500,
    loopGapTimer: null,
    pollTimer: null,
    rafId: null,
    transcriptTab: "pdf",
    selectedAudioFile: null,
    selectedPdfFile: null,
    pausedAtSentenceEnd: false,
    lastSeekMs: 0,
  };

  // DOM helpers
  const $ = (id) => document.getElementById(id);
  const audio = $("audioPlayer");

  const SPEAKER_PALETTE = [
    ["rgba(99,102,241,0.18)", "#a5b4fc"],
    ["rgba(14,165,233,0.18)", "#7dd3fc"],
    ["rgba(16,185,129,0.18)", "#6ee7b7"],
    ["rgba(245,158,11,0.18)", "#fcd34d"],
    ["rgba(236,72,153,0.18)", "#f9a8d4"],
  ];

  function speakerStyle(name) {
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) | 0;
    const [bg, fg] = SPEAKER_PALETTE[Math.abs(hash) % SPEAKER_PALETTE.length];
    return document.body.classList.contains("theme-light")
      ? `background:${bg};color:var(--accent);`
      : `background:${bg};color:${fg};`;
  }

  function formatTime(sec, withTenths = true) {
    if (!Number.isFinite(sec) || sec < 0) sec = 0;
    const mins = Math.floor(sec / 60);
    const rem = sec - mins * 60;
    if (withTenths) {
      const s = rem.toFixed(1).padStart(4, "0");
      return `${String(mins).padStart(2, "0")}:${s}`;
    }
    const s = String(Math.floor(rem)).padStart(2, "0");
    return `${String(mins).padStart(2, "0")}:${s}`;
  }

  function clearLoopTimer() {
    if (state.loopGapTimer) {
      clearTimeout(state.loopGapTimer);
      state.loopGapTimer = null;
    }
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
  }

  // API Helpers
  async function apiJson(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
      let msg = `请求失败 (${res.status})`;
      try {
        const data = await res.json();
        if (data.detail) msg = data.detail;
      } catch (_) {}
      throw new Error(msg);
    }
    return res.json();
  }

  // Views switching
  function showView(viewName) {
    $("uploadView").classList.toggle("hidden", viewName !== "upload");
    $("processingView").classList.toggle("hidden", viewName !== "processing");
    $("lyricsStage").classList.toggle("hidden", viewName !== "player");
    $("playerDock").classList.toggle("hidden", viewName !== "player");
    $("playerTools").classList.toggle("hidden", viewName !== "player");
  }

  // Library Sidebar
  async function loadProjectList() {
    try {
      state.projects = await apiJson("/api/projects");
      renderProjectList();
    } catch (err) {
      console.error(err);
    }
  }

  function renderProjectList() {
    const list = $("projectList");
    if (!state.projects.length) {
      list.innerHTML = `<div class="empty-library">暂无听力项目，点击上方上传</div>`;
      return;
    }
    list.innerHTML = "";
    for (const p of state.projects) {
      const item = document.createElement("div");
      item.className = "project-item" + (state.currentProject?.id === p.id ? " active" : "");
      const statusBadge =
        p.status === "processing"
          ? "对齐中..."
          : p.status === "error"
          ? "失败"
          : `${formatTime(p.duration, false)} · ${p.sentence_count}句`;

      item.innerHTML = `
        <div class="project-item-title">${escapeHtml(p.title)}</div>
        <div class="project-item-meta">
          <span>${statusBadge}</span>
        </div>
        <button class="project-delete-btn" title="删除">✕</button>
      `;

      item.addEventListener("click", (e) => {
        if (e.target.closest(".project-delete-btn")) return;
        $("sidebar").classList.remove("open");
        openProject(p.id);
      });

      item.querySelector(".project-delete-btn").addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`确定删除 "${p.title}" 吗？`)) return;
        try {
          await apiJson(`/api/projects/${p.id}`, { method: "DELETE" });
          if (state.currentProject?.id === p.id) {
            audio.pause();
            audio.src = "";
            state.currentProject = null;
            showView("upload");
            $("trackTitle").textContent = "欢迎使用 ListenSync";
            $("trackSub").textContent = "上传 MP3 音频与 PDF 原文，像看歌词一样听听力、逐句精听";
          }
          await loadProjectList();
        } catch (err) {
          alert(err.message || "删除失败");
        }
      });

      list.appendChild(item);
    }
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  // Open & Render Project
  async function openProject(projectId) {
    stopPolling();
    clearLoopTimer();
    state.soloOverride = null;
    state.pausedAtSentenceEnd = false;
    state.editingSentenceId = null;

    const proj = await apiJson(`/api/projects/${projectId}`);
    state.currentProject = proj;
    renderProjectList();

    $("trackTitle").textContent = proj.title;
    $("trackSub").textContent = `${proj.audio_filename} · ${proj.transcript_filename}`;

    if (proj.status === "processing") {
      showView("processing");
      renderProcessing(proj);
      pollProjectStatus(proj.id);
      return;
    }

    if (proj.status === "error") {
      showView("processing");
      renderProcessing(proj);
      return;
    }

    // Ready -> Setup Player
    loadReadyProject(proj);
  }

  function renderProcessing(proj) {
    const pct = proj.progress ?? 0;
    $("processingTitle").textContent = `正在对齐：${proj.title}`;
    $("processingMessage").textContent = proj.progress_message || "处理中...";
    $("processingFill").style.width = `${pct}%`;
    $("processingPct").textContent = `${pct}%`;

    const hasErr = proj.status === "error";
    document.querySelector(".spinner-ring")?.classList.toggle("hidden", hasErr);
    $("processingErrorBox").classList.toggle("hidden", !hasErr);
    if (hasErr) {
      $("processingErrorText").textContent = proj.error || "处理过程中发生未知错误";
    }
  }

  function pollProjectStatus(projectId) {
    stopPolling();
    state.pollTimer = setTimeout(async () => {
      try {
        const proj = await apiJson(`/api/projects/${projectId}`);
        if (state.currentProject?.id !== projectId) return;
        state.currentProject = proj;

        if (proj.status === "processing") {
          renderProcessing(proj);
          pollProjectStatus(projectId);
        } else if (proj.status === "error") {
          renderProcessing(proj);
          loadProjectList();
        } else {
          loadProjectList();
          loadReadyProject(proj);
        }
      } catch (err) {
        pollProjectStatus(projectId);
      }
    }, 800);
  }

  function loadReadyProject(proj) {
    state.currentProject = proj;
    state.spokenSentences = (proj.sentences || []).filter((s) => s.kind !== "header");
    state.activeSentenceId = state.spokenSentences[0]?.id ?? null;

    showView("player");

    audio.pause();
    audio.src = `/api/projects/${proj.id}/audio?t=${Date.now()}`;
    audio.playbackRate = parseFloat($("speedSelect").value) || 1.0;
    audio.volume = parseFloat($("volumeRange").value) ?? 1.0;

    $("totalTimeLabel").textContent = formatTime(proj.duration || 0);
    $("currentTimeLabel").textContent = "00:00.0";
    updateSentenceCounter();

    renderLyrics();
    drawWaveform();
    startSyncLoop();
  }

  function updateSentenceCounter() {
    const total = state.spokenSentences.length;
    const idx = state.spokenSentences.findIndex((s) => s.id === state.activeSentenceId);
    const num = idx >= 0 ? idx + 1 : 0;
    $("sentenceCounter").textContent = `第 ${num} / ${total} 句`;
  }

  // Lyrics Rendering
  function renderLyrics() {
    const container = $("lyricsList");
    const sentences = state.currentProject?.sentences || [];
    const q = state.searchQuery.trim().toLowerCase();
    container.innerHTML = "";

    let spokenOrdinal = 0;
    let lastSpeaker = "";

    for (const s of sentences) {
      if (s.kind === "header") {
        const div = document.createElement("div");
        div.className = "section-divider";
        div.textContent = s.text;
        div.title = "点击跳转到此章节";
        div.addEventListener("click", () => {
          seekAndPlayTime(s.start);
        });
        container.appendChild(div);
        lastSpeaker = "";
        continue;
      }

      spokenOrdinal += 1;
      const isMatch = !q || s.text.toLowerCase().includes(q) || (s.speaker && s.speaker.toLowerCase().includes(q));
      const row = document.createElement("div");
      row.className = "lyric-row" + (s.id === state.activeSentenceId ? " active" : "") + (isMatch ? "" : " search-dim");
      row.dataset.id = String(s.id);

      const showSpeaker = s.speaker && s.speaker !== lastSpeaker;
      lastSpeaker = s.speaker || "";
      const lowConf = (s.confidence ?? 1) < 0.35;

      const isSoloOnce = state.soloOverride?.id === s.id && state.soloOverride?.mode === "once";
      const isSoloLoop =
        (state.soloOverride?.id === s.id && state.soloOverride?.mode === "loop") ||
        (state.playMode === "loop" && state.activeSentenceId === s.id && !state.soloOverride);

      row.innerHTML = `
        <div class="lyric-time-col">
          <span class="lyric-idx">#${String(spokenOrdinal).padStart(2, "0")}${
            lowConf ? '<span class="conf-dot" title="自动插值边界，可点击右侧微调"></span>' : ""
          }</span>
          <span class="lyric-time">${formatTime(s.start)}</span>
        </div>
        <div class="lyric-body">
          ${
            showSpeaker
              ? `<div class="speaker-badge" style="${speakerStyle(s.speaker)}">${escapeHtml(s.speaker)}</div>`
              : ""
          }
          <div class="lyric-text">${escapeHtml(s.text)}</div>
        </div>
        <div class="lyric-actions">
          <button class="mini-btn btn-solo ${isSoloOnce ? "solo-active" : ""}" title="仅播放这一句，播完自动暂停">▶ 单句</button>
          <button class="mini-btn btn-loop ${isSoloLoop ? "solo-active" : ""}" title="单句循环听这一句">🔁 循环</button>
          <button class="mini-btn btn-edit" title="微调开始/结束时间或修改文本">✎</button>
        </div>
      `;

      // Clicking the sentence body
      row.querySelector(".lyric-body").addEventListener("click", () => {
        if (state.blindMode) {
          row.classList.toggle("revealed");
        }
        onSentenceClicked(s);
      });

      // Solo play once
      row.querySelector(".btn-solo").addEventListener("click", (e) => {
        e.stopPropagation();
        playSentenceSolo(s, "once");
      });

      // Solo loop
      row.querySelector(".btn-loop").addEventListener("click", (e) => {
        e.stopPropagation();
        if (state.soloOverride?.id === s.id && state.soloOverride?.mode === "loop") {
          state.soloOverride = null;
          renderLyrics();
        } else {
          playSentenceSolo(s, "loop");
        }
      });

      // Inline timestamp & text editor
      row.querySelector(".btn-edit").addEventListener("click", (e) => {
        e.stopPropagation();
        state.editingSentenceId = state.editingSentenceId === s.id ? null : s.id;
        renderLyrics();
      });

      if (state.editingSentenceId === s.id) {
        row.appendChild(buildSentenceEditor(s));
      }

      container.appendChild(row);
    }
  }

  function buildSentenceEditor(s) {
    const box = document.createElement("div");
    box.className = "sentence-editor";
    box.innerHTML = `
      <textarea class="editor-text">${escapeHtml(s.text)}</textarea>
      <div class="editor-controls">
        <div class="time-nudge-group">
          <span>开始 <strong>${s.start.toFixed(2)}s</strong></span>
          <button type="button" class="nudge-btn" data-nudge="start:-0.2">-0.2s</button>
          <button type="button" class="nudge-btn" data-nudge="start:0.2">+0.2s</button>
          <button type="button" class="nudge-btn" data-set-now="start">设为当前播放时刻</button>
          <span style="margin-left:8px;">结束 <strong>${s.end.toFixed(2)}s</strong></span>
          <button type="button" class="nudge-btn" data-nudge="end:-0.2">-0.2s</button>
          <button type="button" class="nudge-btn" data-nudge="end:0.2">+0.2s</button>
          <button type="button" class="nudge-btn" data-set-now="end">设为当前播放时刻</button>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button type="button" class="mini-btn" data-action="test">▶ 试听</button>
          <button type="button" class="mini-btn" data-action="split">✂ 居中拆分</button>
          <button type="button" class="mini-btn" data-action="merge">⊕ 合并下句</button>
          <button type="button" class="mini-btn solo-active" data-action="save">保存文本</button>
        </div>
      </div>
    `;

    box.querySelectorAll("[data-nudge]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const [field, deltaStr] = btn.dataset.nudge.split(":");
        const nextVal = Math.max(0, Number(s[field]) + parseFloat(deltaStr));
        const updated = await patchSentence(s.id, { [field]: nextVal });
        playSentenceSolo(updated, "once");
      });
    });

    box.querySelectorAll("[data-set-now]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const field = btn.dataset.setNow;
        await patchSentence(s.id, { [field]: audio.currentTime });
      });
    });

    box.querySelector('[data-action="test"]').addEventListener("click", () => {
      playSentenceSolo(s, "once");
    });

    box.querySelector('[data-action="save"]').addEventListener("click", async () => {
      const txt = box.querySelector(".editor-text").value.trim();
      if (txt) await patchSentence(s.id, { text: txt });
    });

    function applyUpdatedSentences(sentences) {
      state.currentProject.sentences = sentences;
      state.spokenSentences = sentences.filter((x) => x.kind !== "header");
      state.editingSentenceId = null;
      renderLyrics();
      drawWaveform();
    }

    box.querySelector('[data-action="split"]').addEventListener("click", async () => {
      const txt = s.text;
      let mid = Math.floor(txt.length / 2);
      const spacePos = txt.indexOf(" ", mid);
      if (spacePos > 1 && spacePos < txt.length - 1) mid = spacePos;
      try {
        const res = await apiJson(`/api/projects/${state.currentProject.id}/sentences/${s.id}/split`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ split_index: mid }),
        });
        applyUpdatedSentences(res.sentences);
      } catch (err) {
        alert(err.message || "拆分句子失败");
      }
    });

    box.querySelector('[data-action="merge"]').addEventListener("click", async () => {
      try {
        const res = await apiJson(`/api/projects/${state.currentProject.id}/sentences/${s.id}/merge-next`, {
          method: "POST",
        });
        applyUpdatedSentences(res.sentences);
      } catch (err) {
        alert(err.message || "合并句子失败");
      }
    });

    return box;
  }

  async function patchSentence(sentenceId, body) {
    const updated = await apiJson(`/api/projects/${state.currentProject.id}/sentences/${sentenceId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const list = state.currentProject.sentences;
    const idx = list.findIndex((x) => x.id === sentenceId);
    if (idx >= 0) list[idx] = updated;
    state.spokenSentences = list.filter((x) => x.kind !== "header");
    renderLyrics();
    drawWaveform();
    return updated;
  }

  // Playback Logic
  function onSentenceClicked(sentence) {
    clearLoopTimer();
    state.pausedAtSentenceEnd = false;
    $("pauseCue").classList.add("hidden");

    if (state.playMode === "continuous") {
      state.soloOverride = null;
      setActiveSentence(sentence.id, true);
      seekAndPlayTime(sentence.start);
    } else if (state.playMode === "single") {
      playSentenceSolo(sentence, "once");
    } else if (state.playMode === "loop") {
      playSentenceSolo(sentence, "loop");
    }
  }

  function playSentenceSolo(sentence, mode) {
    clearLoopTimer();
    state.pausedAtSentenceEnd = false;
    $("pauseCue").classList.add("hidden");

    state.soloOverride = {
      id: sentence.id,
      start: sentence.start,
      end: sentence.end,
      mode,
    };
    setActiveSentence(sentence.id, true);
    renderLyrics();
    seekAndPlayTime(sentence.start);
  }

  function seekAndPlayTime(sec) {
    const target = Math.max(0, Math.min(state.currentProject?.duration || 99999, sec));
    state.lastSeekMs = performance.now();
    audio.currentTime = target;
    audio.play().catch((err) => console.warn("Audio play blocked:", err));
    updatePlayButton();
  }

  function setActiveSentence(id, scrollIfAuto = false) {
    if (state.activeSentenceId === id && !scrollIfAuto) return;
    const changed = state.activeSentenceId !== id;
    state.activeSentenceId = id;
    updateSentenceCounter();

    if (changed) {
      document.querySelectorAll(".lyric-row.active").forEach((el) => el.classList.remove("active"));
      const row = document.querySelector(`.lyric-row[data-id="${id}"]`);
      if (row) row.classList.add("active");
    }

    if (scrollIfAuto && state.autoScroll && !state.userScrolling) {
      scrollToSentence(id);
    }
    drawWaveform();
  }

  function scrollToSentence(id) {
    const row = document.querySelector(`.lyric-row[data-id="${id}"]`);
    if (row) {
      row.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  function findSentenceAtTime(t) {
    const list = state.spokenSentences;
    if (!list.length) return null;
    // Binary search for last sentence with start <= t + 0.04
    let lo = 0;
    let hi = list.length - 1;
    let best = list[0];
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid].start <= t + 0.04) {
        best = list[mid];
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return best;
  }

  function startSyncLoop() {
    if (state.rafId) cancelAnimationFrame(state.rafId);

    const tick = () => {
      if (state.currentProject && state.currentProject.status === "ready") {
        const t = audio.currentTime || 0;
        $("currentTimeLabel").textContent = formatTime(t);

        if (!audio.paused && !audio.seeking && performance.now() - state.lastSeekMs > 90) {
          // Determine active boundary enforcement
          const activeTarget =
            state.soloOverride ||
            (state.playMode !== "continuous"
              ? state.spokenSentences.find((s) => s.id === state.activeSentenceId)
              : null);

          if (activeTarget) {
            const loopMode = state.soloOverride ? state.soloOverride.mode === "loop" : state.playMode === "loop";
            if (t >= activeTarget.end - 0.015) {
              if (loopMode) {
                if (state.loopGapMs > 0) {
                  audio.pause();
                  updatePlayButton();
                  clearLoopTimer();
                  state.loopGapTimer = setTimeout(() => {
                    if (!state.currentProject) return;
                    state.lastSeekMs = performance.now();
                    audio.currentTime = activeTarget.start;
                    audio.play().catch(() => {});
                    updatePlayButton();
                  }, state.loopGapMs);
                } else {
                  state.lastSeekMs = performance.now();
                  audio.currentTime = activeTarget.start;
                }
              } else {
                // Single play finished
                audio.pause();
                audio.currentTime = activeTarget.end;
                state.soloOverride = null;
                state.pausedAtSentenceEnd = true;
                $("pauseCue").classList.remove("hidden");
                updatePlayButton();
                renderLyrics();
              }
            }
          } else {
            // Continuous mode -> update active sentence automatically
            const s = findSentenceAtTime(t);
            if (s && s.id !== state.activeSentenceId) {
              setActiveSentence(s.id, true);
            }
          }
        }

        drawWaveform();
      }
      state.rafId = requestAnimationFrame(tick);
    };

    state.rafId = requestAnimationFrame(tick);
  }

  function updatePlayButton() {
    $("playPauseBtn").textContent = audio.paused ? "▶" : "⏸";
  }

  function togglePlayPause() {
    clearLoopTimer();
    if (!state.currentProject) return;

    if (!audio.paused) {
      audio.pause();
      updatePlayButton();
      return;
    }

    $("pauseCue").classList.add("hidden");
    // In single-sentence mode, if paused right at end of sentence, advance to next sentence on Space
    if (state.playMode === "single" && state.pausedAtSentenceEnd) {
      state.pausedAtSentenceEnd = false;
      jumpRelativeSentence(1);
      return;
    }

    if (state.playMode === "single" || state.playMode === "loop") {
      const cur =
        state.spokenSentences.find((s) => s.id === state.activeSentenceId) ||
        findSentenceAtTime(audio.currentTime || 0) ||
        state.spokenSentences[0];
      if (cur) {
        // If currentTime is outside cur bounds, start from cur.start
        if (audio.currentTime < cur.start - 0.05 || audio.currentTime >= cur.end - 0.05) {
          audio.currentTime = cur.start;
        }
        setActiveSentence(cur.id, true);
      }
    }

    audio.play().catch(() => {});
    updatePlayButton();
  }

  function jumpRelativeSentence(delta) {
    const list = state.spokenSentences;
    if (!list.length) return;
    let idx = list.findIndex((s) => s.id === state.activeSentenceId);
    if (idx < 0) idx = 0;
    const nextIdx = Math.max(0, Math.min(list.length - 1, idx + delta));
    const target = list[nextIdx];
    onSentenceClicked(target);
  }

  function replayCurrentSentence() {
    const list = state.spokenSentences;
    if (!list.length) return;
    const cur =
      list.find((s) => s.id === state.activeSentenceId) ||
      findSentenceAtTime(audio.currentTime || 0) ||
      list[0];
    if (!cur) return;
    if (state.playMode === "continuous") {
      playSentenceSolo(cur, "once");
    } else {
      onSentenceClicked(cur);
    }
  }

  function setPlayMode(mode) {
    clearLoopTimer();
    state.playMode = mode;
    state.soloOverride = null;
    state.pausedAtSentenceEnd = false;
    $("pauseCue").classList.add("hidden");

    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    $("loopGapSelect").classList.toggle("hidden", mode !== "loop");
    renderLyrics();
  }

  // Waveform Canvas
  function drawWaveform() {
    const canvas = $("waveformCanvas");
    const wrap = $("waveformWrap");
    const proj = state.currentProject;
    if (!canvas || !wrap || !proj) return;

    const dpr = window.devicePixelRatio || 1;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w <= 0 || h <= 0) return;

    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const dur = proj.duration || audio.duration || 1;
    const curTime = audio.currentTime || 0;
    const curRatio = Math.max(0, Math.min(1, curTime / dur));

    // Highlight active sentence span
    const activeSent = state.spokenSentences.find((s) => s.id === state.activeSentenceId);
    if (activeSent) {
      const x0 = (activeSent.start / dur) * w;
      const x1 = (activeSent.end / dur) * w;
      ctx.fillStyle = "rgba(99, 102, 241, 0.18)";
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    }

    // Sentence boundary ticks
    ctx.fillStyle = "rgba(148, 163, 184, 0.22)";
    for (const s of state.spokenSentences) {
      const sx = (s.start / dur) * w;
      ctx.fillRect(sx, 0, 1, 6);
      ctx.fillRect(sx, h - 6, 1, 6);
    }

    // Waveform bars
    const peaks = proj.waveform && proj.waveform.length ? proj.waveform : new Array(160).fill(0.35);
    const barCount = Math.min(peaks.length, Math.floor(w / 3));
    const step = w / barCount;
    const barW = Math.max(1.5, step * 0.62);

    for (let i = 0; i < barCount; i++) {
      const pIdx = Math.floor((i / barCount) * peaks.length);
      const amp = Math.max(0.08, peaks[pIdx]);
      const barH = amp * (h - 12);
      const x = i * step;
      const y = (h - barH) / 2;
      const isPlayed = i / barCount <= curRatio;
      ctx.fillStyle = isPlayed ? "#22d3ee" : "rgba(148, 163, 184, 0.38)";
      ctx.fillRect(x, y, barW, barH);
    }

    // Playhead cursor
    const px = curRatio * w;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(px - 1, 0, 2, h);
  }

  // Event Bindings
  function initEvents() {
    // Theme toggle
    $("themeToggleBtn").addEventListener("click", () => {
      const isLight = document.body.classList.toggle("theme-light");
      $("themeToggleBtn").textContent = isLight ? "☀️" : "🌙";
      renderLyrics();
    });

    $("toggleSidebarBtn").addEventListener("click", () => {
      $("sidebar").classList.toggle("open");
    });

    $("newProjectBtn").addEventListener("click", () => {
      stopPolling();
      clearLoopTimer();
      audio.pause();
      showView("upload");
      $("trackTitle").textContent = "上传新听力";
      $("trackSub").textContent = "上传 MP3 音频与 PDF 原文，自动对齐每句话";
      $("sidebar").classList.remove("open");
    });

    // Transcript tabs (PDF vs paste text)
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.transcriptTab = btn.dataset.tab;
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
        $("pdfDropzone").classList.toggle("hidden", state.transcriptTab !== "pdf");
        $("textPasteBox").classList.toggle("hidden", state.transcriptTab !== "text");
      });
    });

    // Drag & drop helpers
    function bindDropzone(zoneEl, inputEl, onFile) {
      zoneEl.addEventListener("dragover", (e) => {
        e.preventDefault();
        zoneEl.classList.add("drag-over");
      });
      zoneEl.addEventListener("dragleave", () => zoneEl.classList.remove("drag-over"));
      zoneEl.addEventListener("drop", (e) => {
        e.preventDefault();
        zoneEl.classList.remove("drag-over");
        const f = e.dataTransfer?.files?.[0];
        if (f) onFile(f);
      });
      inputEl.addEventListener("change", () => {
        const f = inputEl.files?.[0];
        if (f) onFile(f);
      });
    }

    bindDropzone($("audioDropzone"), $("audioFileInput"), (file) => {
      state.selectedAudioFile = file;
      const pill = $("audioFilePill");
      pill.textContent = `✓ ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
      pill.classList.remove("hidden");
      if (!$("titleInput").value.trim()) {
        $("titleInput").value = file.name.replace(/\.[^.]+$/, "");
      }
    });

    bindDropzone($("pdfDropzone"), $("pdfFileInput"), (file) => {
      state.selectedPdfFile = file;
      const pill = $("pdfFilePill");
      pill.textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
      pill.classList.remove("hidden");
    });

    // Upload Form submit
    $("uploadForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const errBox = $("uploadError");
      errBox.classList.add("hidden");

      if (!state.selectedAudioFile) {
        errBox.textContent = "请先选择听力 MP3 / 音频文件。";
        errBox.classList.remove("hidden");
        return;
      }

      const pasted = $("transcriptTextarea").value.trim();
      if (state.transcriptTab === "pdf" && !state.selectedPdfFile && !pasted) {
        errBox.textContent = "请上传听力原文 PDF 文件（或切换到直接粘贴原文）。";
        errBox.classList.remove("hidden");
        return;
      }
      if (state.transcriptTab === "text" && !pasted) {
        errBox.textContent = "请粘贴听力原文文本。";
        errBox.classList.remove("hidden");
        return;
      }

      const fd = new FormData();
      fd.append("audio", state.selectedAudioFile);
      if (state.transcriptTab === "pdf" && state.selectedPdfFile) {
        fd.append("pdf", state.selectedPdfFile);
      } else {
        fd.append("transcript_text", pasted);
      }
      fd.append("title", $("titleInput").value.trim());
      fd.append("language", $("languageSelect").value);
      fd.append("model_size", $("modelSelect").value);

      const submitBtn = $("submitUploadBtn");
      submitBtn.disabled = true;
      submitBtn.textContent = "正在上传并解析原文...";

      try {
        const res = await apiJson("/api/projects", { method: "POST", body: fd });
        // Reset file selections
        state.selectedAudioFile = null;
        state.selectedPdfFile = null;
        $("audioFileInput").value = "";
        $("pdfFileInput").value = "";
        $("audioFilePill").classList.add("hidden");
        $("pdfFilePill").classList.add("hidden");
        $("titleInput").value = "";

        await loadProjectList();
        await openProject(res.id);
      } catch (err) {
        errBox.textContent = err.message || "上传失败";
        errBox.classList.remove("hidden");
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "开始智能同步对齐";
      }
    });

    // Player controls
    $("playPauseBtn").addEventListener("click", togglePlayPause);
    $("prevSentenceBtn").addEventListener("click", () => jumpRelativeSentence(-1));
    $("nextSentenceBtn").addEventListener("click", () => jumpRelativeSentence(1));
    $("replaySentenceBtn").addEventListener("click", replayCurrentSentence);

    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => setPlayMode(btn.dataset.mode));
    });

    $("loopGapSelect").addEventListener("change", (e) => {
      state.loopGapMs = parseInt(e.target.value, 10) || 0;
    });

    $("speedSelect").addEventListener("change", (e) => {
      audio.playbackRate = parseFloat(e.target.value) || 1.0;
    });

    $("volumeRange").addEventListener("input", (e) => {
      audio.volume = parseFloat(e.target.value);
    });

    audio.addEventListener("play", updatePlayButton);
    audio.addEventListener("pause", updatePlayButton);
    audio.addEventListener("ended", () => {
      updatePlayButton();
      clearLoopTimer();
      const activeTarget =
        state.soloOverride ||
        (state.playMode !== "continuous"
          ? state.spokenSentences.find((s) => s.id === state.activeSentenceId)
          : null);
      const loopMode = state.soloOverride ? state.soloOverride.mode === "loop" : state.playMode === "loop";
      if (activeTarget && loopMode) {
        state.loopGapTimer = setTimeout(() => {
          if (!state.currentProject) return;
          state.lastSeekMs = performance.now();
          audio.currentTime = activeTarget.start;
          audio.play().catch(() => {});
          updatePlayButton();
        }, state.loopGapMs);
      } else if (activeTarget) {
        state.soloOverride = null;
        state.pausedAtSentenceEnd = true;
        $("pauseCue").classList.remove("hidden");
        renderLyrics();
      }
    });

    // Waveform scrubbing
    const wfWrap = $("waveformWrap");
    let scrubbing = false;
    const seekFromPointer = (clientX, playNow = false) => {
      const rect = wfWrap.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const dur = state.currentProject?.duration || audio.duration || 0;
      const target = ratio * dur;
      clearLoopTimer();
      state.soloOverride = null;
      state.pausedAtSentenceEnd = false;
      state.lastSeekMs = performance.now();
      $("pauseCue").classList.add("hidden");
      audio.currentTime = target;
      const s = findSentenceAtTime(target);
      if (s) setActiveSentence(s.id, true);
      if (playNow && audio.paused) {
        audio.play().catch(() => {});
      }
      drawWaveform();
    };

    wfWrap.addEventListener("mousedown", (e) => {
      scrubbing = true;
      seekFromPointer(e.clientX, false);
    });
    window.addEventListener("mousemove", (e) => {
      if (scrubbing) seekFromPointer(e.clientX, false);
    });
    window.addEventListener("mouseup", () => {
      scrubbing = false;
    });
    wfWrap.addEventListener("mousemove", (e) => {
      const rect = wfWrap.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const dur = state.currentProject?.duration || audio.duration || 0;
      const hint = $("waveformHoverTime");
      hint.style.left = `${ratio * 100}%`;
      hint.textContent = formatTime(ratio * dur);
    });

    // Lyrics scroll & auto-scroll pause detection
    const lScroll = $("lyricsScroll");
    let scrollReleaseTimer = null;
    lScroll.addEventListener("wheel", () => {
      state.userScrolling = true;
      $("jumpActiveBtn").classList.toggle("hidden", state.activeSentenceId == null);
      if (scrollReleaseTimer) clearTimeout(scrollReleaseTimer);
      scrollReleaseTimer = setTimeout(() => {
        state.userScrolling = false;
        $("jumpActiveBtn").classList.add("hidden");
      }, 4000);
    });

    $("jumpActiveBtn").addEventListener("click", () => {
      state.userScrolling = false;
      $("jumpActiveBtn").classList.add("hidden");
      if (state.activeSentenceId != null) scrollToSentence(state.activeSentenceId);
    });

    $("autoScrollBtn").addEventListener("click", () => {
      state.autoScroll = !state.autoScroll;
      $("autoScrollBtn").classList.toggle("active", state.autoScroll);
    });

    $("blindModeBtn").addEventListener("click", () => {
      state.blindMode = !state.blindMode;
      $("blindModeBtn").classList.toggle("active", state.blindMode);
      $("lyricsList").classList.toggle("blind-mode", state.blindMode);
      document.querySelectorAll(".lyric-row.revealed").forEach((el) => el.classList.remove("revealed"));
    });

    $("searchInput").addEventListener("input", (e) => {
      state.searchQuery = e.target.value;
      renderLyrics();
    });

    // Export menu
    $("exportMenuBtn").addEventListener("click", (e) => {
      e.stopPropagation();
      $("exportDropdown").classList.toggle("hidden");
    });
    document.addEventListener("click", () => $("exportDropdown").classList.add("hidden"));
    document.querySelectorAll("[data-export]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!state.currentProject) return;
        window.location.href = `/api/projects/${state.currentProject.id}/export?format=${btn.dataset.export}`;
      });
    });

    // Realign modal & retry
    const openRealign = () => {
      if (!state.currentProject) return;
      $("realignLanguage").value = state.currentProject.language || "auto";
      $("realignModel").value = state.currentProject.model_size || "base";
      $("realignModal").classList.remove("hidden");
    };
    $("realignBtn").addEventListener("click", openRealign);
    $("retryRealignBtn").addEventListener("click", openRealign);
    $("cancelRealignBtn").addEventListener("click", () => $("realignModal").classList.add("hidden"));
    $("confirmRealignBtn").addEventListener("click", async () => {
      if (!state.currentProject) return;
      const pid = state.currentProject.id;
      $("realignModal").classList.add("hidden");
      try {
        await apiJson(`/api/projects/${pid}/realign`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            language: $("realignLanguage").value,
            model_size: $("realignModel").value,
          }),
        });
        await openProject(pid);
      } catch (err) {
        alert(err.message || "重新对齐请求失败");
      }
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (!state.currentProject || state.currentProject.status !== "ready") return;

      if (e.code === "Space") {
        e.preventDefault();
        togglePlayPause();
      } else if (e.code === "ArrowLeft" || e.code === "ArrowUp") {
        e.preventDefault();
        jumpRelativeSentence(-1);
      } else if (e.code === "ArrowRight" || e.code === "ArrowDown") {
        e.preventDefault();
        jumpRelativeSentence(1);
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        replayCurrentSentence();
      } else if (e.key === "b" || e.key === "B") {
        e.preventDefault();
        $("blindModeBtn").click();
      } else if (e.key === "1") {
        setPlayMode("continuous");
      } else if (e.key === "2") {
        setPlayMode("single");
      } else if (e.key === "3") {
        setPlayMode("loop");
      }
    });

    window.addEventListener("resize", drawWaveform);
  }

  // Boot
  initEvents();
  loadProjectList().then(() => {
    if (state.projects.length > 0 && state.projects[0].status === "ready") {
      openProject(state.projects[0].id);
    } else {
      showView("upload");
    }
  });
})();
