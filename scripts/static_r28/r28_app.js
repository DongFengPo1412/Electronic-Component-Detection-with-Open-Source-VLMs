/* R29 Workbench Client Logic - Offline Zero-CDN, Slots Contract, Linked Compare Viewports */
(() => {
  const CLS_META = [
    { key: "capacitor", cn: "电容", color: "#0284c7" },
    { key: "resistor", cn: "电阻", color: "#16a34a" },
    { key: "inductor", cn: "电感", color: "#d97706" },
    { key: "transistor", cn: "晶体管", color: "#db2777" },
    { key: "diode", cn: "二极管", color: "#ea580c" },
    { key: "ic", cn: "集成电路", color: "#7c3aed" },
  ];
  const CLS_COLOR = Object.fromEntries(CLS_META.map(c => [c.key, c.color]));
  const CLS_CN = Object.fromEntries(CLS_META.map(c => [c.key, c.cn]));

  const STATE = {
    upload: null,
    method: "composite",
    activeJobId: null,
    activeJobStatus: null,
    activeResult: null,
    activeQuery: null,
    activeCmpId: null,
    activeCmpData: null,
    cmpSlotResults: {},
    cmpGtScores: {},
    cmpGtErrors: {},
    cmpLoadSeq: 0,
    cmpGtEvalSeq: 0,
    pollTimer: null,
    cmpPollTimer: null,
    showBoxes: true,
    coordView: "orig",
    visibleClasses: new Set(CLS_META.map(c => c.key)),
    zoom: 1,
    panX: 0,
    panY: 0,
    imgW: 0,
    imgH: 0,
    previewW: 0,
    previewH: 0,
    cmpZoom: 1,
    cmpPanX: 0,
    cmpPanY: 0,
    cmpShowBoxes: true,
    cmpCoordView: "orig",
    mobileCmpModel: "qwen3",
    researchOverview: null,
  };

  const $ = (id) => document.getElementById(id);

  function fmtSec(val) {
    if (val === null || val === undefined || val === "") return "-";
    const n = Number(val);
    return Number.isFinite(n) ? `${n.toFixed(1)}s` : "-";
  }

  function el(tag, attrs = {}, text = null) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "className") node.className = v;
      else if (k === "style") node.setAttribute("style", v);
      else node.setAttribute(k, v);
    }
    if (text !== null && text !== undefined) node.textContent = String(text);
    return node;
  }

  async function apiJson(url, opts = {}) {
    const resp = await fetch(url, opts);
    const text = await resp.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  function setActiveTab(tabName) {
    const map = { detect: ["tabBtnLive", "panelLive"], compare: ["tabBtnCompare", "panelCompare"], history: ["tabBtnResearch", "panelResearch"] };
    for (const [k, [btnId, panelId]] of Object.entries(map)) {
      const active = k === tabName;
      $(btnId).classList.toggle("active", active);
      $(btnId).setAttribute("aria-selected", active ? "true" : "false");
      $(panelId).classList.toggle("panel-hidden", !active);
    }
    if (tabName === "history") {
      refreshHistoryLists();
      loadResearchOverview();
    }
  }

  function clearStaleResults() {
    if (STATE.pollTimer) { clearTimeout(STATE.pollTimer); STATE.pollTimer = null; }
    STATE.activeJobId = null;
    STATE.activeJobStatus = null;
    STATE.activeResult = null;
    STATE.activeQuery = null;
    $("liveSvg").innerHTML = "";
    $("btnExportJson").disabled = true;
    $("btnExportPng").disabled = true;
    $("btnRunCompositeQuery").disabled = true;
    $("detectStatusBox").style.display = "none";
    $("queryFeedbackBox").style.display = "none";
    $("rawOutputPre").textContent = "暂无原始输出";
    if ($("predCountLabel")) $("predCountLabel").textContent = "整板预测框数（未核真值）";
    if ($("liveGtStatus")) {
      $("liveGtStatus").textContent = "未导入标注，仅展示模型预测框数（不代表板上实际器件数）。";
      $("liveGtStatus").style.color = "var(--muted)";
    }
    if ($("liveGtResultBox")) {
      $("liveGtResultBox").style.display = "none";
      $("liveGtResultBox").innerHTML = "";
    }
    if ($("btnEvalLiveGt")) $("btnEvalLiveGt").disabled = true;
    renderClassCounts({});
  }

  function setMethod(method) {
    STATE.method = method;
    $("btnMethodComposite").classList.toggle("active", method === "composite");
    $("btnMethodVlm").classList.toggle("active", method === "vlm");
    const badge = $("methodExecBadge");
    if (method === "composite") {
      badge.className = "badge b-ok";
      badge.textContent = "筛选已有预测（CPU · 0 GPU）";
      $("btnRunPrimary").textContent = "启动日常检测（R15 YOLO）";
      $("queryModeHint").textContent = "在当前 R15 预测结果上做类别/区域筛选与计数（CPU · 0 GPU）";
      $("vlmInstructionBox").style.display = "none";
      $("compositeQueryBox").style.display = "block";
    } else {
      badge.className = "badge b-info";
      badge.textContent = "按指令重新推理（独立 GPU 调用）";
      $("btnRunPrimary").textContent = "启动纯 VLM 检测（Qwen3 R5/R8）";
      $("queryModeHint").textContent = "纯 VLM 新指令独立推理（支持单类指令；将触发新 GPU 任务）";
      $("vlmInstructionBox").style.display = "block";
      $("compositeQueryBox").style.display = "none";
    }
  }

  function renderClassCounts(counts) {
    const grid = $("clsCountGrid");
    grid.innerHTML = "";
    for (const c of CLS_META) {
      const n = Number((counts && counts[c.key]) || 0);
      const pill = el("button", { type: "button", className: "cls-pill" + (STATE.visibleClasses.has(c.key) ? "" : " off") });
      const left = el("span", { className: "row", style: "gap:5px;" });
      left.appendChild(el("span", { className: "cls-dot", style: `background:${c.color}` }));
      left.appendChild(el("span", {}, c.cn));
      pill.appendChild(left);
      pill.appendChild(el("strong", {}, String(n)));
      pill.onclick = () => {
        if (STATE.visibleClasses.has(c.key)) STATE.visibleClasses.delete(c.key);
        else STATE.visibleClasses.add(c.key);
        syncClassCheckboxes();
        renderOverlay();
      };
      grid.appendChild(pill);
    }
  }

  function syncClassCheckboxes() {
    document.querySelectorAll(".cls-filter-chk").forEach(chk => {
      chk.checked = STATE.visibleClasses.has(chk.value);
    });
    document.querySelectorAll("#clsCountGrid .cls-pill").forEach((pill, i) => {
      pill.classList.toggle("off", !STATE.visibleClasses.has(CLS_META[i].key));
    });
  }

  function applyTransform() {
    $("liveStage").style.transform = `translate(${STATE.panX}px, ${STATE.panY}px) scale(${STATE.zoom})`;
    $("zoomPctLbl").textContent = `${Math.round(STATE.zoom * 100)}%`;
  }

  function fitCanvasToWindow() {
    const pw = STATE.previewW || STATE.imgW;
    const ph = STATE.previewH || STATE.imgH;
    if (!pw || !ph) return;
    const wrap = $("liveCanvasWrap");
    const cw = Math.max(200, wrap.clientWidth - 16);
    const ch = Math.max(200, wrap.clientHeight - 16);
    const s = Math.min(cw / pw, ch / ph, 1.0);
    STATE.zoom = Math.max(0.05, s);
    STATE.panX = Math.max(0, (wrap.clientWidth - pw * STATE.zoom) / 2);
    STATE.panY = Math.max(0, (wrap.clientHeight - ph * STATE.zoom) / 2);
    applyTransform();
  }

  function renderSvgBoxes(svg, preds, imgW, imgH, ox, oy, showBoxes, visibleSet) {
    svg.innerHTML = "";
    if (!imgW || !imgH) return 0;
    svg.setAttribute("viewBox", `0 0 ${imgW} ${imgH}`);
    if (!showBoxes) return 0;
    let shown = 0;
    const sw = Math.max(2, Math.round(imgW / 900)).toString();
    for (const p of (preds || [])) {
      const lbl = String(p.label || p.aligned_fics6_label || "").toLowerCase();
      if (visibleSet && !visibleSet.has(lbl)) continue;
      const box = p.bbox_xyxy;
      if (!box || box.length !== 4) continue;
      const x1 = Number(box[0]) - ox;
      const y1 = Number(box[1]) - oy;
      const w = Math.max(2, Number(box[2]) - Number(box[0]));
      const h = Math.max(2, Number(box[3]) - Number(box[1]));
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x1.toFixed(1));
      rect.setAttribute("y", y1.toFixed(1));
      rect.setAttribute("width", w.toFixed(1));
      rect.setAttribute("height", h.toFixed(1));
      rect.setAttribute("fill", "none");
      rect.setAttribute("stroke", CLS_COLOR[lbl] || "#38bdf8");
      rect.setAttribute("stroke-width", sw);
      svg.appendChild(rect);
      shown++;
    }
    return shown;
  }

  function renderOverlay() {
    const svg = $("liveSvg");
    const pw = STATE.previewW || STATE.imgW;
    const ph = STATE.previewH || STATE.imgH;
    svg.style.width = `${pw}px`;
    svg.style.height = `${ph}px`;
    const res = STATE.activeResult;
    if (!res) { svg.innerHTML = ""; return; }
    const preds = STATE.activeQuery ? (STATE.activeQuery.predictions || []) : (res.predictions || []);
    const roi = res.roi || { offset_x: 0, offset_y: 0 };
    const ox = STATE.coordView === "roi" ? Number(roi.offset_x || 0) : 0;
    const oy = STATE.coordView === "roi" ? Number(roi.offset_y || 0) : 0;
    const shown = renderSvgBoxes(svg, preds, STATE.imgW, STATE.imgH, ox, oy, STATE.showBoxes, STATE.visibleClasses);
    $("visibleBoxStats").textContent = `当前显示 ${shown} / ${preds.length} 个预测框`;
  }

  function loadPreviewImage(uploadId, roi) {
    const img = $("liveImg");
    const srcMode = STATE.coordView === "roi" ? "roi" : "orig";
    const targetW = STATE.coordView === "roi" && roi ? Number(roi.width) : Number(STATE.upload.orig_wh[0]);
    const targetH = STATE.coordView === "roi" && roi ? Number(roi.height) : Number(STATE.upload.orig_wh[1]);
    const previewW = Math.min(targetW, 1600);
    const previewH = Math.max(1, Math.round(targetH * (previewW / Math.max(1, targetW))));
    STATE.imgW = targetW;
    STATE.imgH = targetH;
    STATE.previewW = previewW;
    STATE.previewH = previewH;
    img.style.width = `${previewW}px`;
    img.style.height = `${previewH}px`;
    img.onload = () => {
      $("canvasEmptyMsg").style.display = "none";
      fitCanvasToWindow();
      renderOverlay();
    };
    img.src = `/api/upload_preview/${encodeURIComponent(uploadId)}?source=${srcMode}&max_width=1600`;
  }

  async function handleFileUpload(file) {
    if (!file) return;
    clearStaleResults();
    STATE.cmpLoadSeq += 1;
    STATE.cmpGtEvalSeq += 1;
    $("uploadErrBox").style.display = "none";
    $("dropZoneSub").textContent = `正在上传并预检：${file.name}...`;
    try {
      const up = await apiJson("/api/upload", {
        method: "POST",
        headers: { "X-Filename": encodeURIComponent(file.name) },
        body: file,
      });
      STATE.upload = up;
      const roi = up.roi || {
        width: up.roi_width,
        height: up.roi_height,
        offset_x: up.roi_offset_x || 0,
        offset_y: up.roi_offset_y || 0,
        tile_count: up.tile_count || 0,
      };
      up.roi = roi;
      $("dropZone").classList.add("dropzone-compact");
      $("dropZoneTitle").textContent = `已载入：${up.safe_filename}`;
      $("dropZoneSub").textContent = `原图尺寸 ${up.orig_wh[0]}×${up.orig_wh[1]} · ROI ${roi.width}×${roi.height} · ${roi.tile_count} 切片`;
      $("btnRunPrimary").disabled = false;
      $("btnStartCompositeJob").disabled = false;
      $("btnStartSingleJob").disabled = false;
      $("btnStartInstructionJob").disabled = false;
      $("btnStartExpSpatialJob").disabled = false;
      $("btnRunThreeCompare").disabled = false;
      $("cmpUploadInfo").textContent = `当前对比图像：${up.safe_filename} (${up.orig_wh[0]}×${up.orig_wh[1]})`;
      loadPreviewImage(up.upload_id, roi);
    } catch (err) {
      $("uploadErrBox").textContent = `上传失败：${err.message}`;
      $("uploadErrBox").style.display = "block";
      $("dropZoneSub").textContent = "支持 PNG / JPG / TIFF，点击或拖拽上传";
    }
  }

  function updateJobStatusUI(st) {
    STATE.activeJobStatus = st;
    $("detectStatusBox").style.display = "block";
    const status = st.status || "unknown";
    const pct = st.base_calls_planned ? Math.min(100, Math.round((Number(st.base_calls_completed || 0) / Number(st.base_calls_planned)) * 100)) : (status === "completed" ? 100 : 15);
    $("jobProgressFill").style.width = `${pct}%`;
    $("jobStageLbl").textContent = st.stage_label || status;
    $("jobTimeLbl").textContent = fmtSec(st.elapsed_seconds);

    const badge = $("statusStateBadge");
    $("btnCancelJob").style.display = (status === "running" || status === "queued") ? "inline-block" : "none";

    if (status === "completed") {
      badge.className = "badge b-ok";
      badge.textContent = "已完成";
    } else if (status === "failed") {
      badge.className = "badge b-err";
      badge.textContent = `失败：${st.error_message || "运行异常"}`;
    } else if (status === "canceled") {
      badge.className = "badge b-warn";
      badge.textContent = "已取消";
    } else {
      badge.className = "badge b-info";
      badge.textContent = "处理中";
    }
  }

  function renderLiveGtResult(rawPhe) {
    const box = $("liveGtResultBox");
    const st = $("liveGtStatus");
    const lbl = $("predCountLabel");
    if (!box || !st) return;
    const phe = unwrapPostHocEval(rawPhe);
    if (!phe) {
      box.style.display = "none";
      box.innerHTML = "";
      if (lbl && !STATE.activeQuery) lbl.textContent = "整板预测框数（未核真值）";
      st.textContent = "未导入标注，仅展示模型预测框数（不代表板上实际器件数）。";
      st.style.color = "var(--muted)";
      return;
    }
    const m = phe.metrics || {};
    const isPartial = phe.eval_scope === "partial_windows" || phe.whole_board_f1_valid === false;
    const winCnt = Array.isArray(phe.eval_windows) ? phe.eval_windows.length : 0;
    const tp = fmtFiniteInt(m.tp);
    const fp = fmtFiniteInt(m.fp);
    const fn = fmtFiniteInt(m.fn);
    const pStr = fmtPct2(m.precision, m.precision_display);
    const rStr = fmtPct2(m.recall, m.recall_display);
    const f1Str = fmtPct2(m.f1, m.f1_display);
    const predTot = fmtFiniteInt(m.total_predictions);
    const gtTot = fmtFiniteInt(m.total_gt);
    const diffStr = fmtSignedDiff(m.total_predictions, m.total_gt);
    if (lbl && !STATE.activeQuery) lbl.textContent = "整板预测框数";
    st.textContent = isPartial
      ? `局部已核窗口评分完成（覆盖 ${winCnt} 窗口 / ${gtTot} GT · 不产生整板 F1 · 0 GPU）`
      : "整图 COCO GT IoU50 评分已完成（0 GPU）。";
    st.style.color = isPartial ? "var(--warn)" : "var(--ok)";
    box.innerHTML = "";
    if (isPartial) {
      box.appendChild(el("span", { className: "badge b-warn", style: "margin-bottom:4px;" }, "局部已核窗口评分（非整板 F1）"));
      const winIds = (phe.eval_windows || []).map(w => w.window_id).join(", ");
      box.appendChild(el("div", { style: "font-size:11px;color:var(--muted);margin-bottom:4px;" }, `覆盖窗口（${winCnt} 个）：${winIds}`));
    } else {
      box.appendChild(el("span", { className: "badge b-ok", style: "margin-bottom:4px;" }, "整图标注评分"));
    }
    box.appendChild(el("div", {}, `IoU50 TP/FP/FN=${tp}/${fp}/${fn} · P=${pStr} R=${rStr} F1=${f1Str}`));
    box.appendChild(el("div", { style: "font-size:11.5px;color:var(--muted);" }, `${isPartial ? "窗口内预测" : "整图预测"}=${predTot} vs 实际标注=${gtTot}（计数差 ${diffStr}）`));
    const idBadgeCls = phe.image_identity_verified ? "b-ok" : "b-warn";
    box.appendChild(el("span", { className: `badge ${idBadgeCls}`, style: "margin-top:4px;" }, formatIdentityBasis(phe)));
    box.style.display = "block";
    if ($("liveGtDrawer")) $("liveGtDrawer").open = true;
  }

  async function evaluateLiveJobWithCocoGt() {
    const file = $("fileLiveGtCoco") && $("fileLiveGtCoco").files && $("fileLiveGtCoco").files[0];
    const jid = STATE.activeJobId;
    if (!file || !jid) return;
    $("liveGtStatus").textContent = "正在对齐 COCO GT 计算 IoU50 指标（0 GPU）…";
    $("liveGtStatus").style.color = "var(--muted)";
    try {
      const coco = JSON.parse(await file.text());
      const ev = await apiJson(`/api/jobs/${encodeURIComponent(jid)}/evaluate_gt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ coco_gt: coco }),
      });
      if (STATE.activeJobId !== jid) return;
      STATE.activeResult = ev;
      renderLiveGtResult(ev.post_hoc_eval);
    } catch (err) {
      if (STATE.activeJobId !== jid) return;
      const hasValidPrev = Boolean(STATE.activeResult && unwrapPostHocEval(STATE.activeResult.post_hoc_eval));
      $("liveGtStatus").textContent = `标注评分失败：${err.message}` + (hasValidPrev ? "（已保留上一次有效评分）" : "");
      $("liveGtStatus").style.color = "var(--err)";
    }
  }

  function applyCompletedResult(res) {
    STATE.activeResult = res;
    STATE.activeQuery = null;
    const preds = res.predictions || [];
    const counts = res.class_counts || {};
    renderClassCounts(counts);
    $("totalPredCount").textContent = String(preds.length);
    if ($("predCountLabel")) $("predCountLabel").textContent = unwrapPostHocEval(res.post_hoc_eval) ? "整板预测框数" : "整板预测框数（未核真值）";
    if (res.roi && res.roi.tile_count && $("dropZoneSub")) {
      const ow = (res.orig_wh || [])[0] || res.roi.width;
      const oh = (res.orig_wh || [])[1] || res.roi.height;
      const ts = res.roi.tile_size || 640;
      $("dropZoneSub").textContent = `原图 ${ow}×${oh} · ROI ${res.roi.width}×${res.roi.height} · 本次检测实际 ${ts}px 切片 ${res.roi.tile_count} 片`;
    }
    $("btnExportJson").disabled = false;
    $("btnExportPng").disabled = false;
    $("btnRunCompositeQuery").disabled = !(res.mode === "COMPOSITE_YOLO" || res.mode === "COMPOSITE_YOLO_QWEN3");
    if ($("btnEvalLiveGt")) {
      const hasLiveFile = Boolean($("fileLiveGtCoco") && $("fileLiveGtCoco").files && $("fileLiveGtCoco").files[0]);
      $("btnEvalLiveGt").disabled = !hasLiveFile;
    }
    renderLiveGtResult(res.post_hoc_eval);
    $("rawOutputPre").textContent = JSON.stringify({
      job_id: res.job_id,
      model_key: res.model_key,
      mode: res.mode,
      detector_preset: res.detector_preset || (res.composite_metadata && res.composite_metadata.detector_preset),
      elapsed_seconds: res.elapsed_seconds,
      class_counts: counts,
      truncation_or_empty: res.outcome || null,
    }, null, 2);

    const outcome = res.outcome || {};
    const badge = $("statusStateBadge");
    if (outcome.outcome_code === "truncated" || res.any_truncated || (outcome.outcome_note && outcome.outcome_note.includes("上限"))) {
      badge.className = "badge b-warn";
      badge.textContent = "已截断（达到生成上限，结果可能不完整）";
    } else if (outcome.zero_box_reason === "format_non_compliant_zero_valid_boxes" || outcome.outcome_code === "parse_failed") {
      badge.className = "badge b-err";
      badge.textContent = "解析失败（输出格式不合规）";
    } else if (preds.length === 0) {
      badge.className = "badge b-warn";
      badge.textContent = "模型显式空 []";
    }
    loadPreviewImage(res.upload_id, res.roi);
  }

  async function pollJob(jobId) {
    try {
      const st = await apiJson(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (STATE.activeJobId !== jobId) return;
      updateJobStatusUI(st);
      if (st.status === "completed") {
        const res = await apiJson(`/api/jobs/${encodeURIComponent(jobId)}/result`);
        if (STATE.activeJobId !== jobId) return;
        applyCompletedResult(res);
        refreshHistoryLists();
        return;
      }
      if (st.status === "failed" || st.status === "canceled") {
        refreshHistoryLists();
        return;
      }
      STATE.pollTimer = setTimeout(() => pollJob(jobId), 1200);
    } catch (err) {
      $("statusStateBadge").className = "badge b-err";
      $("statusStateBadge").textContent = `状态查询失败：${err.message}`;
    }
  }

  async function startJob(endpoint, payload) {
    if (!STATE.upload) return;
    clearStaleResults();
    try {
      const sub = await apiJson(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      STATE.activeJobId = sub.job_id;
      updateJobStatusUI({ status: "queued", stage_label: "已提交任务，等待执行…", base_calls_planned: 1, base_calls_completed: 0 });
      pollJob(sub.job_id);
    } catch (err) {
      $("detectStatusBox").style.display = "block";
      $("statusStateBadge").className = "badge b-err";
      $("statusStateBadge").textContent = `提交失败：${err.message}`;
    }
  }

  async function runCompositeQuery(qText) {
    const box = $("queryFeedbackBox");
    box.style.display = "none";
    if (!STATE.activeJobId || !STATE.activeResult) {
      box.className = "notice n-warn";
      box.textContent = "请先完成一次 R15 日常检测，再执行筛选或空间查询。";
      box.style.display = "block";
      return;
    }
    if (!String(qText || "").trim()) {
      box.className = "notice n-err";
      box.textContent = "请输入查询指令（例如：只找电阻，但只统计左半边）。";
      box.style.display = "block";
      return;
    }
    try {
      const q = await apiJson("/api/composite/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: STATE.activeJobId, query_text: qText }),
      });
      STATE.activeQuery = q;
      const cnt = Number(q.predicted_count || 0);
      const wholeCnt = Array.isArray(STATE.activeResult.predictions) ? STATE.activeResult.predictions.length : cnt;
      if ($("predCountLabel")) $("predCountLabel").textContent = "当前查询筛选框数";
      $("totalPredCount").textContent = String(cnt);
      renderClassCounts(q.class_counts || {});
      box.className = cnt === 0 ? "notice n-warn" : "notice n-ok";
      box.textContent = cnt === 0
        ? `当前查询 0 框（整板预测共 ${wholeCnt} 框）｜机制：${q.execution_path_label || "筛选已生成的检测结果（0 GPU）"}`
        : `当前查询 ${cnt} 框（区域外排除 ${q.class_matched_excluded_by_region || 0}；整板预测共 ${wholeCnt} 框）｜${q.execution_path_label || "筛选已生成的检测结果（0 GPU）"}`;
      box.style.display = "block";
      renderOverlay();
    } catch (err) {
      box.className = "notice n-err";
      box.textContent = `当前主线不支持该查询或参数有误：${err.message}`;
      box.style.display = "block";
    }
  }

  // ===== Compare Tab (Reads cmp.slots + Linked Viewports + Post-Hoc GT) =====
  function applyCmpTransforms() {
    document.querySelectorAll(".cmp-stage").forEach(st => {
      st.style.transform = `translate(${STATE.cmpPanX}px, ${STATE.cmpPanY}px) scale(${STATE.cmpZoom})`;
    });
    $("cmpZoomPctLbl").textContent = `${Math.round(STATE.cmpZoom * 100)}%`;
  }

  function fitCmpViewports() {
    const cmp = STATE.activeCmpData;
    if (!cmp) return;
    const roi = cmp.roi || {};
    const tw = STATE.cmpCoordView === "roi" && roi.width ? Number(roi.width) : Number((cmp.orig_wh || [768])[0]);
    const th = STATE.cmpCoordView === "roi" && roi.height ? Number(roi.height) : Number((cmp.orig_wh || [768, 768])[1]);
    const pw = Math.min(tw, 900);
    const ph = Math.max(1, Math.round(th * (pw / Math.max(1, tw))));
    const vp = document.querySelector(".cmp-viewport");
    const cw = vp ? Math.max(180, vp.clientWidth - 12) : 340;
    const ch = vp ? Math.max(180, vp.clientHeight - 12) : 268;
    const s = Math.min(cw / pw, ch / ph, 1.0);
    STATE.cmpZoom = Math.max(0.05, s);
    STATE.cmpPanX = Math.max(0, (cw + 12 - pw * STATE.cmpZoom) / 2);
    STATE.cmpPanY = Math.max(0, (ch + 12 - ph * STATE.cmpZoom) / 2);
    applyCmpTransforms();
  }

  function renderAllCmpSlotOverlays() {
    const cmp = STATE.activeCmpData;
    if (!cmp) return;
    const roi = cmp.roi || {};
    const tw = STATE.cmpCoordView === "roi" && roi.width ? Number(roi.width) : Number((cmp.orig_wh || [768])[0]);
    const th = STATE.cmpCoordView === "roi" && roi.height ? Number(roi.height) : Number((cmp.orig_wh || [768, 768])[1]);
    const pw = Math.min(tw, 900);
    const ph = Math.max(1, Math.round(th * (pw / Math.max(1, tw))));
    const ox = STATE.cmpCoordView === "roi" ? Number(roi.offset_x || 0) : 0;
    const oy = STATE.cmpCoordView === "roi" ? Number(roi.offset_y || 0) : 0;

    for (const slot of (cmp.slots || [])) {
      const mk = slot.model_key;
      const vp = document.querySelector(`.cmp-viewport[data-m="${mk}"]`);
      if (!vp) continue;
      const stage = vp.querySelector(".cmp-stage");
      const img = vp.querySelector("img");
      const svg = vp.querySelector("svg");
      stage.style.width = `${pw}px`;
      stage.style.height = `${ph}px`;
      img.style.width = `${pw}px`;
      img.style.height = `${ph}px`;
      svg.style.width = `${pw}px`;
      svg.style.height = `${ph}px`;
      const srcMode = STATE.cmpCoordView === "roi" ? "roi" : "orig";
      const wantedSrc = `/api/upload_preview/${encodeURIComponent(cmp.upload_id)}?source=${srcMode}&max_width=900`;
      if (img.getAttribute("data-src") !== wantedSrc) {
        img.setAttribute("data-src", wantedSrc);
        img.src = wantedSrc;
      }
      const jobRes = STATE.cmpSlotResults[mk];
      const preds = jobRes ? (jobRes.predictions || []) : [];
      renderSvgBoxes(svg, preds, tw, th, ox, oy, STATE.cmpShowBoxes, null);
    }
    applyCmpTransforms();
  }

  function unwrapPostHocEval(raw) {
    if (!raw || typeof raw !== "object") return null;
    const phe = (raw.post_hoc_eval && typeof raw.post_hoc_eval === "object") ? raw.post_hoc_eval : raw;
    if (!phe || phe.gt_evaluated !== true || !phe.metrics || typeof phe.metrics !== "object") return null;
    const m = phe.metrics;
    if (!Number.isFinite(Number(m.tp)) || !Number.isFinite(Number(m.fp)) || !Number.isFinite(Number(m.fn))) return null;
    return phe;
  }

  function fmtFiniteInt(v) {
    if (v === null || v === undefined || v === "") return "N/A";
    const n = Number(v);
    return Number.isFinite(n) ? String(Math.round(n)) : "N/A";
  }

  function fmtPct2(v, fallbackDisplay) {
    if (v === null || v === undefined || v === "") return fallbackDisplay || "N/A";
    const n = Number(v);
    return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : (fallbackDisplay || "N/A");
  }

  function fmtSignedDiff(predCnt, gtCnt) {
    const p = Number(predCnt);
    const g = Number(gtCnt);
    if (!Number.isFinite(p) || !Number.isFinite(g)) return "N/A";
    const d = Math.round(p) - Math.round(g);
    return d >= 0 ? `+${d}` : String(d);
  }

  function formatIdentityBasis(phe) {
    if (!phe || typeof phe !== "object") return "未知身份依据";
    if (phe.image_identity_verified === true || phe.identity_basis === "sha256_match") {
      return "SHA-256 字节校验通过";
    }
    if (phe.identity_basis === "filename_and_dimensions_only") {
      return "仅文件名+尺寸匹配（未验证 SHA-256）";
    }
    if (phe.identity_basis === "manual_user_confirmation") {
      return "人工确认映射（未验证字节）";
    }
    return String(phe.identity_basis || "未知身份依据");
  }

  function extractPersistedGtScoresFromCmp(cmp) {
    for (const slot of (cmp.slots || [])) {
      const mk = String(slot.model_key || "");
      if (!mk) continue;
      const fromSlot = unwrapPostHocEval(slot.post_hoc_eval);
      const fromRes = unwrapPostHocEval(STATE.cmpSlotResults[mk]);
      const phe = fromSlot || fromRes;
      if (phe) {
        STATE.cmpGtScores[mk] = phe;
        delete STATE.cmpGtErrors[mk];
      }
    }
  }

  function updateCmpGtStatusBanner(cmp, opts = {}) {
    const elStatus = $("cmpGtEvalStatus");
    if (!elStatus) return;
    const completed = (cmp && Array.isArray(cmp.slots) ? cmp.slots : []).filter(s => s.queue_status === "completed" && s.job_id);
    const scoredCnt = completed.filter(s => Boolean(unwrapPostHocEval(STATE.cmpGtScores[s.model_key]))).length;
    const errEntries = Object.entries(STATE.cmpGtErrors || {});
    if (opts.explicitText) {
      elStatus.textContent = opts.explicitText;
      elStatus.style.color = opts.color || "var(--muted)";
      return;
    }
    if (errEntries.length > 0 && scoredCnt > 0) {
      elStatus.textContent = `部分槽位评分完成（${scoredCnt}/${completed.length}），部分失败：${errEntries.map(([k, v]) => `${k}:${v}`).join("；")}`;
      elStatus.style.color = "var(--warn)";
    } else if (errEntries.length > 0) {
      elStatus.textContent = `标注评分失败：${errEntries.map(([k, v]) => `${k}:${v}`).join("；")}`;
      elStatus.style.color = "var(--err)";
    } else if (completed.length > 0 && scoredCnt === completed.length) {
      elStatus.textContent = opts.restored
        ? "已恢复事后 COCO GT IoU50 评分（持久化读回 · 0 新增 GPU）。"
        : "COCO GT IoU50 评分已完成（0 新增 GPU）。";
      elStatus.style.color = "var(--ok)";
    } else if (scoredCnt > 0) {
      elStatus.textContent = `部分槽位已有事后 COCO GT 评分（${scoredCnt}/${completed.length}）。`;
      elStatus.style.color = "var(--warn)";
    } else {
      elStatus.textContent = "未导入标注，仅展示模型预测与运行状态。";
      elStatus.style.color = "var(--muted)";
    }
  }

  async function hydrateCmpSlotResults(cmp, loadSeq) {
    const slots = cmp.slots || [];
    let fetchedAny = false;
    await Promise.all(slots.map(async (slot) => {
      if (slot.queue_status === "completed" && slot.job_id && !STATE.cmpSlotResults[slot.model_key]) {
        try {
          const res = await apiJson(`/api/jobs/${encodeURIComponent(slot.job_id)}/result`);
          if (loadSeq !== undefined && loadSeq !== STATE.cmpLoadSeq) return;
          if (STATE.activeCmpId !== cmp.comparison_id) return;
          STATE.cmpSlotResults[slot.model_key] = res;
          fetchedAny = true;
        } catch { /* keep non-blocking */ }
      }
    }));
    if (loadSeq !== undefined && loadSeq !== STATE.cmpLoadSeq) return;
    if (STATE.activeCmpId !== cmp.comparison_id) return;
    extractPersistedGtScoresFromCmp(cmp);
    updateCmpGtStatusBanner(cmp, { restored: true });
    renderComparisonColumns(cmp);
    if (fetchedAny) fitCmpViewports();
  }

  function renderComparisonColumns(cmp) {
    const slots = Array.isArray(cmp.slots) ? cmp.slots : [];
    const totalSec = cmp.wall_elapsed_seconds !== undefined && cmp.wall_elapsed_seconds !== null
      ? cmp.wall_elapsed_seconds
      : cmp.sum_job_elapsed_seconds;
    $("cmpStatusNote").textContent = `对比状态：${cmp.status || "-"} ｜ 图像：${cmp.safe_filename || "-"} ｜ 总耗时：${fmtSec(totalSec)}`;
    if (cmp.safe_filename) $("cmpUploadInfo").textContent = `当前对比图像：${cmp.safe_filename}`;

    const container = $("cmpColumns");
    container.innerHTML = "";
    let hasCompletedSlot = false;

    for (const slot of slots) {
      const mk = String(slot.model_key || "");
      const qStatus = String(slot.queue_status || "waiting");
      if (qStatus === "completed" && slot.job_id) hasCompletedSlot = true;

      const col = el("div", {
        className: "card cmp-col" + (STATE.mobileCmpModel === mk ? "" : " mobile-hidden"),
        "data-model": mk,
      });

      const titleRow = el("div", { className: "card-title" });
      titleRow.appendChild(el("span", {}, `${mk.toUpperCase()} (${slot.model_display_name || mk})`));
      const stCls = qStatus === "completed" ? "b-ok" : qStatus === "failed" ? "b-err" : qStatus === "canceled" ? "b-warn" : "b-info";
      titleRow.appendChild(el("span", { className: `badge ${stCls}` }, qStatus));
      col.appendChild(titleRow);

      // Synchronized same-image viewport
      const vp = el("div", { className: "cmp-viewport", "data-m": mk });
      const stage = el("div", { className: "cmp-stage" });
      stage.appendChild(el("img", { alt: `${mk} 叠框预览` }));
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "cmp-svg");
      stage.appendChild(svg);
      vp.appendChild(stage);
      let dragStart = null;
      vp.onpointerdown = (e) => { dragStart = { x: e.clientX - STATE.cmpPanX, y: e.clientY - STATE.cmpPanY }; vp.classList.add("dragging"); };
      vp.onpointermove = (e) => { if (!dragStart) return; STATE.cmpPanX = e.clientX - dragStart.x; STATE.cmpPanY = e.clientY - dragStart.y; applyCmpTransforms(); };
      vp.onpointerup = () => { dragStart = null; vp.classList.remove("dragging"); };
      vp.onpointerleave = () => { dragStart = null; vp.classList.remove("dragging"); };
      col.appendChild(vp);

      // Aligned metadata table (safe textContent)
      const tbl = el("table", { className: "tbl" });
      const addRow = (k, vNode) => {
        const tr = el("tr");
        tr.appendChild(el("th", {}, k));
        const td = el("td");
        if (typeof vNode === "string" || typeof vNode === "number") td.textContent = String(vNode);
        else td.appendChild(vNode);
        tr.appendChild(td);
        tbl.appendChild(tr);
      };

      const outcome = slot.outcome || {};
      const cnt = slot.kept_predictions_count !== null && slot.kept_predictions_count !== undefined ? slot.kept_predictions_count : "-";
      addRow("配置", `${slot.preset || cmp.preset || "baseline"} (${slot.mode || cmp.mode || "W"})`);
      addRow("耗时", fmtSec(slot.elapsed_seconds));
      addRow("预测数量", el("strong", {}, String(cnt)));

      // Status/failure/truncation reason
      const reasonWrap = el("div", { className: "stack", style: "gap:4px;" });
      if (qStatus === "failed") {
        reasonWrap.appendChild(el("span", { className: "badge b-err" }, `失败：${slot.error_message || "运行失败"}`));
      } else if (outcome.zero_box_reason === "format_non_compliant_zero_valid_boxes") {
        reasonWrap.appendChild(el("span", { className: "badge b-err" }, "无有效框（输出格式不合规）"));
      } else {
        reasonWrap.appendChild(el("span", {}, outcome.outcome_label || qStatus));
      }
      if (outcome.outcome_note && (outcome.outcome_note.includes("上限") || outcome.outcome_note.includes("截断"))) {
        reasonWrap.appendChild(el("span", { className: "badge b-warn" }, "达到生成上限，结果可能不完整"));
      }
      addRow("结束原因", reasonWrap);

      const cc = slot.class_counts || {};
      const ccStr = CLS_META.map(c => `${c.cn}:${Number(cc[c.key] || 0)}`).join(" ");
      addRow("逐类分布", ccStr);

      // Post-hoc COCO GT metrics if imported or restored
      const gtErr = STATE.cmpGtErrors[mk];
      const phe = unwrapPostHocEval(STATE.cmpGtScores[mk]);
      if (gtErr) {
        const errWrap = el("div", { className: "stack", style: "gap:4px;" });
        errWrap.appendChild(el("span", { className: "badge b-err" }, `评分失败：${gtErr}`));
        addRow("GT 评分", errWrap);
      } else if (phe) {
        const m = phe.metrics || {};
        const tp = fmtFiniteInt(m.tp);
        const fp = fmtFiniteInt(m.fp);
        const fn = fmtFiniteInt(m.fn);
        const pStr = fmtPct2(m.precision, m.precision_display);
        const rStr = fmtPct2(m.recall, m.recall_display);
        const f1Str = fmtPct2(m.f1, m.f1_display);
        const predTot = fmtFiniteInt(m.total_predictions);
        const gtTot = fmtFiniteInt(m.total_gt);
        const diffStr = fmtSignedDiff(m.total_predictions, m.total_gt);
        const idText = formatIdentityBasis(phe);
        const idBadgeCls = phe.image_identity_verified ? "b-ok" : "b-warn";

        const gtWrap = el("div", { className: "stack", style: "gap:4px;" });
        gtWrap.appendChild(el("div", { className: "gt-score-summary" }, `IoU50 TP/FP/FN=${tp}/${fp}/${fn} · P=${pStr} R=${rStr} F1=${f1Str}`));
        gtWrap.appendChild(el("div", { className: "gt-count-diff", style: "font-size:11.5px;color:var(--muted);" }, `预测/GT=${predTot}/${gtTot}（计数差 ${diffStr}）`));
        const idRow = el("div", { className: "row", style: "gap:4px;font-size:11px;" });
        idRow.appendChild(el("span", { className: `badge ${idBadgeCls}` }, idText));
        if (phe.coco_file_name) {
          idRow.appendChild(el("span", { style: "color:var(--muted);" }, String(phe.coco_file_name)));
        }
        gtWrap.appendChild(idRow);

        const perClass = m.per_class || {};
        const pcDrawer = el("details", { className: "drawer gt-per-class-drawer", style: "margin-top:2px;padding:4px 6px;" });
        pcDrawer.appendChild(el("summary", {}, "逐类 TP/FP/FN 与计数差"));
        const pcTbl = el("table", { className: "tbl", style: "font-size:11px;margin-top:4px;" });
        const pcHead = el("tr");
        for (const h of ["类别", "GT", "预测", "差值", "TP/FP/FN", "F1"]) pcHead.appendChild(el("th", {}, h));
        pcTbl.appendChild(pcHead);
        for (const c of CLS_META) {
          const cm = perClass[c.key] || {};
          const tr = el("tr");
          tr.appendChild(el("td", {}, c.cn));
          tr.appendChild(el("td", {}, fmtFiniteInt(cm.gt_count)));
          tr.appendChild(el("td", {}, fmtFiniteInt(cm.pred_count)));
          tr.appendChild(el("td", {}, fmtSignedDiff(cm.pred_count, cm.gt_count)));
          tr.appendChild(el("td", {}, `${fmtFiniteInt(cm.tp)}/${fmtFiniteInt(cm.fp)}/${fmtFiniteInt(cm.fn)}`));
          const cF1 = Number(cm.gt_count) === 0 && Number(cm.pred_count) === 0 ? "N/A (GT=0)" : fmtPct2(cm.f1, cm.f1_display);
          tr.appendChild(el("td", {}, cF1));
          pcTbl.appendChild(tr);
        }
        pcDrawer.appendChild(pcTbl);
        gtWrap.appendChild(pcDrawer);
        addRow("GT 评分", gtWrap);
      } else {
        addRow("GT 评分", el("span", { style: "color:var(--muted);" }, "未导入标注"));
      }
      col.appendChild(tbl);

      // Action row + raw calls drawer
      const actRow = el("div", { className: "row", style: "margin-top:8px;" });
      const expBtn = el("button", { type: "button", className: "btn-sm" }, "导出单模型 JSON");
      expBtn.disabled = !(qStatus === "completed" && slot.job_id);
      expBtn.onclick = () => { if (slot.job_id) window.location.href = `/api/jobs/${encodeURIComponent(slot.job_id)}/export/json`; };
      actRow.appendChild(expBtn);
      col.appendChild(actRow);

      const rawDrawer = el("details", { className: "drawer" });
      rawDrawer.appendChild(el("summary", {}, "原始调用与输出"));
      const jobRes = STATE.cmpSlotResults[mk];
      const rawPre = el("pre", { style: "font-size:11px;white-space:pre-wrap;word-break:break-all;max-height:140px;overflow:auto;margin:6px 0 0;" });
      if (jobRes && Array.isArray(jobRes.calls) && jobRes.calls.length) {
        rawPre.textContent = JSON.stringify(jobRes.calls.map(c => ({
          call_id: c.call_id,
          finish_reason: c.finish_reason,
          generated_tokens: c.generated_tokens,
          raw_output: String(c.raw_output || "").slice(0, 500),
        })), null, 2);
      } else {
        rawPre.textContent = slot.error_message || "暂无原始调用明细";
      }
      rawDrawer.appendChild(rawPre);
      col.appendChild(rawDrawer);

      container.appendChild(col);
    }

    $("btnEvalCmpGt").disabled = !hasCompletedSlot || !($("fileCmpGtCoco").files && $("fileCmpGtCoco").files[0]);
    renderAllCmpSlotOverlays();
  }

  async function startThreeModelCompare() {
    if (!STATE.upload) return;
    if (STATE.cmpPollTimer) { clearTimeout(STATE.cmpPollTimer); STATE.cmpPollTimer = null; }
    const loadSeq = ++STATE.cmpLoadSeq;
    STATE.cmpGtEvalSeq += 1;
    STATE.cmpSlotResults = {};
    STATE.cmpGtScores = {};
    STATE.cmpGtErrors = {};
    $("cmpStatusNote").textContent = "正在串行执行 Qwen3 / Qwen2 / DeepSeek 三模型对比…";
    updateCmpGtStatusBanner(null);
    $("btnExportCmpJson").disabled = true;
    try {
      const sub = await apiJson("/api/comparisons", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_id: STATE.upload.upload_id,
          model_keys: ["qwen3", "qwen2", "deepseek"],
          mode: $("selCmpMode").value || "W",
          preset: $("selCmpPreset").value || "baseline",
        }),
      });
      if (loadSeq !== STATE.cmpLoadSeq) return;
      STATE.activeCmpId = sub.comparison_id;
      pollComparison(sub.comparison_id, loadSeq);
    } catch (err) {
      $("cmpStatusNote").textContent = `对比启动失败：${err.message}`;
    }
  }

  async function pollComparison(cmpId, loadSeq) {
    const seq = loadSeq !== undefined ? loadSeq : STATE.cmpLoadSeq;
    try {
      const cmp = await apiJson(`/api/comparisons/${encodeURIComponent(cmpId)}`);
      if (seq !== STATE.cmpLoadSeq || STATE.activeCmpId !== cmpId) return;
      STATE.activeCmpData = cmp;
      extractPersistedGtScoresFromCmp(cmp);
      renderComparisonColumns(cmp);
      if (cmp.status === "completed" || cmp.status === "failed" || cmp.status === "canceled") {
        $("btnExportCmpJson").disabled = false;
        await hydrateCmpSlotResults(cmp, seq);
        refreshHistoryLists();
        return;
      }
      STATE.cmpPollTimer = setTimeout(() => pollComparison(cmpId, seq), 1500);
    } catch (err) {
      if (seq !== STATE.cmpLoadSeq) return;
      $("cmpStatusNote").textContent = `对比轮询失败：${err.message}`;
    }
  }

  async function evaluateCmpWithCocoGt() {
    const file = $("fileCmpGtCoco").files && $("fileCmpGtCoco").files[0];
    const cmp = STATE.activeCmpData;
    if (!file || !cmp) return;
    const evalSeq = ++STATE.cmpGtEvalSeq;
    const cmpId = cmp.comparison_id;
    updateCmpGtStatusBanner(cmp, { explicitText: "正在对齐 COCO GT 计算 IoU50 指标（0 GPU）…", color: "var(--muted)" });
    let coco = null;
    try {
      coco = JSON.parse(await file.text());
    } catch (parseErr) {
      if (evalSeq !== STATE.cmpGtEvalSeq || STATE.activeCmpId !== cmpId) return;
      updateCmpGtStatusBanner(cmp, { explicitText: `标注评分失败：COCO GT 文件不是合法 JSON（${parseErr.message}）`, color: "var(--err)" });
      return;
    }

    if (evalSeq !== STATE.cmpGtEvalSeq || STATE.activeCmpId !== cmpId) return;
    STATE.cmpGtScores = {};
    STATE.cmpGtErrors = {};
    let okCount = 0;
    let failCount = 0;
    for (const slot of (cmp.slots || [])) {
      if (slot.queue_status === "completed" && slot.job_id) {
        const mk = slot.model_key;
        try {
          const ev = await apiJson(`/api/jobs/${encodeURIComponent(slot.job_id)}/evaluate_gt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ coco_gt: coco }),
          });
          if (evalSeq !== STATE.cmpGtEvalSeq || STATE.activeCmpId !== cmpId) return;
          const phe = unwrapPostHocEval(ev);
          if (!phe) {
            STATE.cmpGtErrors[mk] = "后端返回评分结构不完整";
            failCount += 1;
          } else {
            STATE.cmpGtScores[mk] = phe;
            okCount += 1;
          }
        } catch (slotErr) {
          if (evalSeq !== STATE.cmpGtEvalSeq || STATE.activeCmpId !== cmpId) return;
          STATE.cmpGtErrors[mk] = slotErr.message || "请求失败";
          failCount += 1;
        }
      }
    }
    if (evalSeq !== STATE.cmpGtEvalSeq || STATE.activeCmpId !== cmpId) return;
    if (okCount === 0 && failCount === 0) {
      updateCmpGtStatusBanner(cmp, { explicitText: "当前对比没有已完成的模型槽位，无法评分。", color: "var(--warn)" });
    } else {
      updateCmpGtStatusBanner(cmp, { restored: false });
    }
    renderComparisonColumns(cmp);
  }

  async function refreshHistoryLists() {
    try {
      const jData = await apiJson("/api/jobs");
      const selJ = $("selRecentJob");
      selJ.innerHTML = `<option value="">（选择本地实时任务 · 共 ${(jData.jobs || []).length} 条）</option>`;
      for (const j of (jData.jobs || [])) {
        const opt = el("option", { value: j.job_id }, `${j.job_id} | ${j.safe_filename} | ${j.model_key}/${j.mode} [${j.status} · 预测数量=${j.kept_predictions_count ?? "-"}]`);
        selJ.appendChild(opt);
      }
      const cData = await apiJson("/api/comparisons");
      const selC = $("selRecentCmp");
      selC.innerHTML = `<option value="">（选择近期三模型对比 · 共 ${(cData.comparisons || []).length} 条）</option>`;
      for (const c of (cData.comparisons || [])) {
        const opt = el("option", { value: c.comparison_id }, `${c.comparison_id} | ${c.safe_filename || ""} [${c.status} · ${fmtSec(c.wall_elapsed_seconds)}]`);
        selC.appendChild(opt);
      }
    } catch { /* ignore */ }
  }

  async function loadResearchOverview() {
    if (STATE.researchOverview) return;
    try {
      const ov = await apiJson("/api/research/overview");
      STATE.researchOverview = ov;
      const selDs = $("selResearchDataset");
      const selB = $("selResearchBoard");
      if (!ov.available) {
        $("researchSummary").textContent = ov.notice || "未加载 8 板只读归档。";
        return;
      }
      selDs.innerHTML = "";
      for (const [dk, ds] of Object.entries(ov.datasets || {})) {
        selDs.appendChild(el("option", { value: dk }, ds.label || dk));
      }
      const updateBoards = () => {
        const ds = (ov.datasets || {})[selDs.value] || {};
        selB.innerHTML = "";
        for (const b of (ds.boards || [])) {
          selB.appendChild(el("option", { value: b.board_id }, `${b.board_id} (GT=${b.gt_count})`));
        }
        if (selDs.value === "dev_two_board_s18_s28") $("selResearchMode").value = "T";
      };
      selDs.onchange = updateBoards;
      updateBoards();
    } catch {
      $("researchSummary").textContent = "加载 8 板研究回放失败。";
    }
  }

  async function loadSelectedResearchBoard() {
    const ds = $("selResearchDataset").value;
    const bid = $("selResearchBoard").value;
    const mk = $("selResearchModel").value;
    const md = $("selResearchMode").value;
    if (!ds || !bid) return;
    try {
      const view = await apiJson(`/api/research/board?dataset=${encodeURIComponent(ds)}&board=${encodeURIComponent(bid)}&model=${encodeURIComponent(mk)}&mode=${encodeURIComponent(md)}`);
      const m = view.metrics || {};
      $("researchSummary").textContent = `${bid} | ${mk} [${md}] | 预测数量=${(view.predictions || []).length} | TP=${m.tp ?? "-"} FP=${m.fp ?? "-"} FN=${m.fn ?? "-"} | F1=${m.f1 != null ? (m.f1 * 100).toFixed(2) + "%" : "-"}`;
      $("researchImg").src = `/api/research/image?dataset=${encodeURIComponent(ds)}&board=${encodeURIComponent(bid)}&max_width=1400`;
    } catch (err) {
      $("researchSummary").textContent = `读取回放失败：${err.message}`;
    }
  }

  function wireEvents() {
    $("tabBtnLive").onclick = () => setActiveTab("detect");
    $("tabBtnCompare").onclick = () => setActiveTab("compare");
    $("tabBtnResearch").onclick = () => setActiveTab("history");

    $("btnMethodComposite").onclick = () => setMethod("composite");
    $("btnMethodVlm").onclick = () => setMethod("vlm");

    const dz = $("dropZone");
    const fb = $("fileBoard");
    dz.onclick = (e) => { if (e.target !== fb) fb.click(); };
    dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fb.click(); } };
    dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("dragover"); };
    dz.ondragleave = () => dz.classList.remove("dragover");
    dz.ondrop = (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
    };
    fb.onchange = () => { if (fb.files && fb.files[0]) handleFileUpload(fb.files[0]); };

    $("btnRunPrimary").onclick = () => {
      if (STATE.method === "composite") $("btnStartCompositeJob").click();
      else $("btnStartSingleJob").click();
    };
    $("btnStartCompositeJob").onclick = () => startJob("/api/jobs", {
      upload_id: STATE.upload.upload_id,
      model_key: $("selCompositeMode").value === "COMPOSITE_YOLO_QWEN3" ? "yolo11n_qwen3_composite" : "yolo11n_composite",
      mode: $("selCompositeMode").value,
      detector_preset: $("selCompositeDetectorPreset").value,
    });
    $("btnStartSingleJob").onclick = () => startJob("/api/jobs", {
      upload_id: STATE.upload.upload_id,
      model_key: $("selSingleModel").value,
      mode: $("selModelMode").value,
      preset: ($("livePreset") || $("selModelPreset")).value,
    });
    $("btnStartInstructionJob").onclick = () => startJob("/api/jobs/instruct", {
      upload_id: STATE.upload.upload_id,
      model_key: $("selSingleModel").value,
      mode: $("selModelMode").value,
      preset: ($("livePreset") || $("selModelPreset")).value,
      user_instruction: $("inpLangInstruction").value,
    });
    $("btnStartExpSpatialJob").onclick = () => startJob("/api/experimental/spatial_job", {
      upload_id: STATE.upload.upload_id,
      model_key: "qwen3",
      mode: "T",
      user_instruction: $("inpExpSpatialInstruction").value,
    });
    $("btnCancelJob").onclick = async () => {
      if (STATE.activeJobId) await apiJson(`/api/jobs/${encodeURIComponent(STATE.activeJobId)}/cancel`, { method: "POST" });
    };

    $("btnRunCompositeQuery").onclick = () => runCompositeQuery($("inpCompositeQuery").value);
    document.querySelectorAll(".btn-comp-q-quick").forEach(b => {
      b.onclick = () => { $("inpCompositeQuery").value = b.getAttribute("data-q") || ""; runCompositeQuery($("inpCompositeQuery").value); };
    });
    document.querySelectorAll(".btn-inst-quick").forEach(b => {
      b.onclick = () => { $("inpLangInstruction").value = b.getAttribute("data-inst") || ""; };
    });

    $("btnZoomIn").onclick = () => { STATE.zoom = Math.min(4, STATE.zoom * 1.25); applyTransform(); };
    $("btnZoomOut").onclick = () => { STATE.zoom = Math.max(0.05, STATE.zoom / 1.25); applyTransform(); };
    $("btnZoomFit").onclick = () => fitCanvasToWindow();
    $("btnToggleBoxes").onclick = () => {
      STATE.showBoxes = !STATE.showBoxes;
      $("btnToggleBoxes").textContent = STATE.showBoxes ? "隐藏框" : "显示框";
      renderOverlay();
    };
    $("selCoordView").onchange = () => {
      STATE.coordView = $("selCoordView").value;
      if (STATE.upload) loadPreviewImage(STATE.upload.upload_id, STATE.upload.roi);
    };
    const wrap = $("liveCanvasWrap");
    let dragStart = null;
    wrap.onpointerdown = (e) => { dragStart = { x: e.clientX - STATE.panX, y: e.clientY - STATE.panY }; wrap.classList.add("dragging"); };
    window.addEventListener("pointermove", (e) => { if (!dragStart) return; STATE.panX = e.clientX - dragStart.x; STATE.panY = e.clientY - dragStart.y; applyTransform(); });
    window.addEventListener("pointerup", () => { dragStart = null; wrap.classList.remove("dragging"); });
    wrap.onwheel = (e) => { e.preventDefault(); STATE.zoom = Math.max(0.05, Math.min(4, STATE.zoom * (e.deltaY < 0 ? 1.12 : 0.89))); applyTransform(); };

    document.querySelectorAll(".cls-filter-chk").forEach(chk => {
      chk.onchange = () => {
        if (chk.checked) STATE.visibleClasses.add(chk.value);
        else STATE.visibleClasses.delete(chk.value);
        syncClassCheckboxes();
        renderOverlay();
      };
    });

    $("btnExportJson").onclick = () => {
      if (!STATE.activeJobId) return;
      window.location.href = (STATE.activeQuery && STATE.activeQuery.query_id)
        ? `/api/composite/query_export/${encodeURIComponent(STATE.activeJobId)}/${encodeURIComponent(STATE.activeQuery.query_id)}`
        : `/api/jobs/${encodeURIComponent(STATE.activeJobId)}/export/json`;
    };
    $("btnExportPng").onclick = () => {
      if (!STATE.activeJobId) return;
      window.location.href = (STATE.activeQuery && STATE.activeQuery.query_id)
        ? `/api/composite/query_overlay/${encodeURIComponent(STATE.activeJobId)}/${encodeURIComponent(STATE.activeQuery.query_id)}`
        : `/api/jobs/${encodeURIComponent(STATE.activeJobId)}/export/overlay_png`;
    };

    // Compare controls
    $("btnRunThreeCompare").onclick = startThreeModelCompare;
    $("btnExportCmpJson").onclick = () => {
      if (STATE.activeCmpId) window.location.href = `/api/comparisons/${encodeURIComponent(STATE.activeCmpId)}/export/json`;
    };
    $("btnCmpZoomIn").onclick = () => { STATE.cmpZoom = Math.min(4, STATE.cmpZoom * 1.25); applyCmpTransforms(); };
    $("btnCmpZoomOut").onclick = () => { STATE.cmpZoom = Math.max(0.05, STATE.cmpZoom / 1.25); applyCmpTransforms(); };
    $("btnCmpZoomFit").onclick = () => fitCmpViewports();
    $("btnCmpToggleBoxes").onclick = () => {
      STATE.cmpShowBoxes = !STATE.cmpShowBoxes;
      $("btnCmpToggleBoxes").textContent = STATE.cmpShowBoxes ? "隐藏框" : "显示框";
      renderAllCmpSlotOverlays();
    };
    $("selCmpCoordView").onchange = () => {
      STATE.cmpCoordView = $("selCmpCoordView").value;
      renderAllCmpSlotOverlays();
      fitCmpViewports();
    };
    $("fileCmpGtCoco").onchange = () => {
      const hasFile = Boolean($("fileCmpGtCoco").files && $("fileCmpGtCoco").files[0]);
      $("btnEvalCmpGt").disabled = !hasFile || !STATE.activeCmpData;
    };
    $("btnEvalCmpGt").onclick = evaluateCmpWithCocoGt;
    if ($("fileLiveGtCoco")) {
      $("fileLiveGtCoco").onchange = () => {
        const hasFile = Boolean($("fileLiveGtCoco").files && $("fileLiveGtCoco").files[0]);
        if ($("btnEvalLiveGt")) $("btnEvalLiveGt").disabled = !hasFile || !STATE.activeJobId || !STATE.activeResult;
      };
    }
    if ($("btnEvalLiveGt")) $("btnEvalLiveGt").onclick = evaluateLiveJobWithCocoGt;
    $("selCmpResultView").onchange = () => {
      const v = $("selCmpResultView").value;
      if (v === "r28_tuned_readonly") {
        $("cmpColumns").classList.add("panel-hidden");
        const rv = $("cmpReadonlyView");
        rv.classList.remove("panel-hidden");
        rv.innerHTML = "";
        rv.appendChild(el("div", { className: "card-title" }, "R28 各自调优只读锁盘参考（来源：R28 S23/S25 4 裁片锁盘实验配置 · 0 新增 GPU）"));
        rv.appendChild(el("p", { style: "font-size:12px;color:var(--muted);margin:4px 0 8px;" }, "注：以下为 R28 在 S23/S25 4 张裁片（82 GT）上的离线锁盘对照，不与当前实时上传图混淆。"));
        const tbl = el("table", { className: "tbl" });
        tbl.innerHTML = "<tr><th>模型</th><th>配置</th><th>TP/FP/FN</th><th>F1</th><th>类无关@50</th><th>备注</th></tr>"
          + "<tr><td>qwen2</td><td>baseline</td><td>0/24/82</td><td>0.00%</td><td>0.00% (0)</td><td>开发集 native_grounding 未超 baseline</td></tr>"
          + "<tr><td>deepseek</td><td>baseline</td><td>0/0/82</td><td>0.00%</td><td>0.00% (0)</td><td>4/4 解析失败</td></tr>"
          + "<tr><td>qwen3</td><td>qwen3_r5_compact_512</td><td>14/44/68</td><td>20.00%</td><td>37.80% (31)</td><td>R/C 互错导致类敏感 F1 未超 R5</td></tr>";
        rv.appendChild(tbl);
      } else {
        $("cmpReadonlyView").classList.add("panel-hidden");
        $("cmpColumns").classList.remove("panel-hidden");
      }
    };

    document.querySelectorAll(".cmp-mob-btn").forEach(b => {
      b.onclick = () => {
        STATE.mobileCmpModel = b.getAttribute("data-m");
        document.querySelectorAll(".cmp-mob-btn").forEach(x => x.classList.toggle("active", x === b));
        document.querySelectorAll("#cmpColumns .cmp-col").forEach(c => {
          c.classList.toggle("mobile-hidden", c.getAttribute("data-model") !== STATE.mobileCmpModel);
        });
      };
    });

    $("btnLoadRecentJob").onclick = async () => {
      const jid = $("selRecentJob").value;
      if (!jid) return;
      try {
        const res = await apiJson(`/api/jobs/${encodeURIComponent(jid)}/result`);
        const up = await apiJson(`/api/uploads/${encodeURIComponent(res.upload_id)}`);
        STATE.upload = up;
        STATE.activeJobId = jid;
        $("dropZone").classList.add("dropzone-compact");
        $("dropZoneTitle").textContent = `已载入：${up.safe_filename}`;
        $("dropZoneSub").textContent = `原图 ${up.orig_wh[0]}×${up.orig_wh[1]} · 历史读回（0 新增 GPU）`;
        $("btnRunPrimary").disabled = false;
        $("btnRunThreeCompare").disabled = false;
        setActiveTab("detect");
        updateJobStatusUI({ status: "completed", stage_label: "历史任务已读回（0 新增 GPU）", elapsed_seconds: res.elapsed_seconds });
        applyCompletedResult(res);
      } catch (err) {
        alert(`载入任务失败：${err.message}`);
      }
    };
    $("btnLoadRecentCmp").onclick = async () => {
      const cid = $("selRecentCmp").value;
      if (!cid) return;
      if (STATE.cmpPollTimer) { clearTimeout(STATE.cmpPollTimer); STATE.cmpPollTimer = null; }
      const loadSeq = ++STATE.cmpLoadSeq;
      STATE.cmpGtEvalSeq += 1;
      STATE.activeCmpId = cid;
      STATE.activeCmpData = null;
      STATE.cmpSlotResults = {};
      STATE.cmpGtScores = {};
      STATE.cmpGtErrors = {};
      try {
        const cmp = await apiJson(`/api/comparisons/${encodeURIComponent(cid)}`);
        if (loadSeq !== STATE.cmpLoadSeq || STATE.activeCmpId !== cid) return;
        STATE.activeCmpData = cmp;
        extractPersistedGtScoresFromCmp(cmp);
        setActiveTab("compare");
        updateCmpGtStatusBanner(cmp, { restored: true });
        renderComparisonColumns(cmp);
        $("btnExportCmpJson").disabled = false;
        if (cmp.status === "completed" || cmp.status === "failed" || cmp.status === "canceled") {
          await hydrateCmpSlotResults(cmp, loadSeq);
        } else {
          pollComparison(cid, loadSeq);
        }
      } catch (err) {
        if (loadSeq !== STATE.cmpLoadSeq) return;
        $("cmpStatusNote").textContent = `载入对比失败：${err.message}`;
      }
    };
    $("btnLoadResearchBoard").onclick = loadSelectedResearchBoard;
  }

  document.addEventListener("DOMContentLoaded", () => {
    wireEvents();
    setMethod("composite");
    renderClassCounts({});
    refreshHistoryLists();
  });
})();
