"""Interactive HTTP server and clean web UI for FICS-PCB final GT-free demo and 8-board research replay."""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import socket
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    CLASS_CN_NAMES,
    DemoError,
    LiveDemoService,
    ResearchReplayStore,
)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:," />
<title>PCB 元器件检测工作台</title>
<link rel="stylesheet" href="/static_r28/r28_app.css" />
</head>
<body>
<header class="topbar">
  <div class="brand">
    <h1>PCB 元器件检测工作台</h1>
    <div class="sub">FICS-PCB 6 类目标（电容 / 电阻 / 电感 / 晶体管 / 二极管 / 集成电路） · 离线本地推理</div>
  </div>
  <nav class="nav-tabs" role="tablist" aria-label="主导航">
    <button id="tabBtnLive" class="tab-btn active" type="button" role="tab" aria-selected="true">检测</button>
    <button id="tabBtnCompare" class="tab-btn" type="button" role="tab" aria-selected="false">三模型对比</button>
    <button id="tabBtnResearch" class="tab-btn" type="button" role="tab" aria-selected="false">历史结果</button>
  </nav>
</header>

<main class="app-main">
  <!-- Tab 1: 检测 -->
  <section id="panelLive" role="tabpanel">
    <div id="dropZone" class="dropzone" tabindex="0" role="button" aria-label="上传 PCB 图片">
      <div id="dropZoneTitle" class="dropzone-title">上传 PCB 图片</div>
      <div id="dropZoneSub" class="dropzone-sub">支持 PNG / JPG / TIFF，点击、拖拽或按回车选择文件</div>
      <input id="fileBoard" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp" style="display:none;" />
    </div>
    <div id="uploadErrBox" class="notice n-err" style="display:none;"></div>

    <div class="detect-grid">
      <!-- 主视区：原图与检测框 -->
      <div class="card">
        <div class="viewport-bar">
          <button id="btnZoomIn" class="btn-sm" type="button" title="放大">+</button>
          <button id="btnZoomOut" class="btn-sm" type="button" title="缩小">-</button>
          <span id="zoomPctLbl" style="min-width:42px;text-align:center;color:var(--muted);">100%</span>
          <button id="btnZoomFit" class="btn-sm" type="button">适应窗口</button>
          <button id="btnToggleBoxes" class="btn-sm" type="button">隐藏框</button>
          <select id="selCoordView" aria-label="坐标基准">
            <option value="orig" selected>原图坐标</option>
            <option value="roi">ROI 裁切视图</option>
          </select>
          <span id="visibleBoxStats" style="margin-left:auto;color:var(--muted);">等待上传图片</span>
        </div>
        <div id="liveCanvasWrap" class="canvas-wrap">
          <div id="liveStage" class="stage">
            <img id="liveImg" alt="PCB 检测原图" />
            <svg id="liveSvg"></svg>
          </div>
          <div id="canvasEmptyMsg" class="canvas-empty">请先在上方上传 PCB 图片，系统将自动预检尺寸与切片</div>
        </div>
      </div>

      <!-- 右侧操作与结果栏 -->
      <div class="stack">
        <div class="card">
          <div class="card-title">
            <span>检测方式</span>
            <span id="methodExecBadge" class="badge b-ok">筛选已有预测（CPU · 0 GPU）</span>
          </div>
          <div class="seg-group">
            <button id="btnMethodComposite" class="seg-btn active" type="button">日常检测（默认 R15）</button>
            <button id="btnMethodVlm" class="seg-btn" type="button">纯 VLM（Qwen3 R5/R8）</button>
          </div>
          <button id="btnRunPrimary" class="btn-primary" type="button" disabled style="width:100%;">启动日常检测（R15 YOLO）</button>

          <div id="detectStatusBox" style="display:none;margin-top:8px;">
            <div class="row" style="justify-content:space-between;">
              <span id="statusStateBadge" class="badge b-info">处理中</span>
              <span id="jobTimeLbl" style="font-size:12px;color:var(--muted);">-</span>
              <button id="btnCancelJob" class="btn-danger btn-sm" type="button" style="display:none;">取消</button>
            </div>
            <div class="progress-track"><div id="jobProgressFill" class="progress-fill"></div></div>
            <div id="jobStageLbl" style="font-size:12px;color:var(--muted);">-</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>类别与预测数量</span>
            <span><span id="predCountLabel">预测框数（未核真值）</span> <strong id="totalPredCount">0</strong></span>
          </div>
          <div id="clsCountGrid" class="cls-grid"></div>
          <div class="row" style="margin-top:8px;font-size:11.5px;color:var(--muted);">
            <label><input class="cls-filter-chk" type="checkbox" value="capacitor" checked /> 电容</label>
            <label><input class="cls-filter-chk" type="checkbox" value="resistor" checked /> 电阻</label>
            <label><input class="cls-filter-chk" type="checkbox" value="inductor" checked /> 电感</label>
            <label><input class="cls-filter-chk" type="checkbox" value="transistor" checked /> 晶体管</label>
            <label><input class="cls-filter-chk" type="checkbox" value="diode" checked /> 二极管</label>
            <label><input class="cls-filter-chk" type="checkbox" value="ic" checked /> 集成电路</label>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span>自然语言查询与导出</span></div>
          <div id="queryModeHint" style="font-size:11.5px;color:var(--muted);margin-bottom:6px;">
            在当前 R15 预测结果上做类别/区域筛选与计数（CPU · 0 GPU）
          </div>
          <div id="compositeQueryBox">
            <div class="row">
              <input id="inpCompositeQuery" type="text" style="flex:1;" value="只找电阻，但只统计左半边" placeholder="例如：只找电容并计数 / 只找电阻，但只统计左半边" />
              <button id="btnRunCompositeQuery" type="button" disabled>筛选查询</button>
            </div>
            <div class="row" style="margin-top:6px;">
              <button type="button" class="btn-sm btn-comp-q-quick" data-q="只找电容并计数">只找电容</button>
              <button type="button" class="btn-sm btn-comp-q-quick" data-q="找电阻和电容">找电阻和电容</button>
              <button type="button" class="btn-sm btn-comp-q-quick" data-q="只找电阻，但只统计左半边">左半边电阻</button>
              <button type="button" class="btn-sm btn-comp-q-quick" data-q="只找电感并计数">只找电感</button>
            </div>
          </div>
          <div id="vlmInstructionBox" style="display:none;">
            <div class="row">
              <input id="inpLangInstruction" type="text" style="flex:1;" value="只找电阻并计数" placeholder="例如：只找电阻并计数" />
              <button id="btnStartInstructionJob" class="btn-primary btn-sm" type="button" disabled>按指令重新推理</button>
            </div>
            <div class="row" style="margin-top:6px;">
              <button type="button" class="btn-sm btn-inst-quick" data-inst="只找电阻并计数">只找电阻</button>
              <button type="button" class="btn-sm btn-inst-quick" data-inst="只找电容并计数">只找电容</button>
            </div>
          </div>
          <div id="queryFeedbackBox" class="notice" style="display:none;"></div>
          <div class="row" style="margin-top:10px;">
            <button id="btnExportJson" type="button" disabled style="flex:1;">导出 JSON</button>
            <button id="btnExportPng" type="button" disabled style="flex:1;">导出叠框 PNG</button>
          </div>
          <details id="liveGtDrawer" class="drawer" style="margin-top:8px;">
            <summary>事后 COCO GT 评分（支持整图或局部已核窗口 · 0 GPU）</summary>
            <div class="stack" style="margin-top:6px;">
              <input id="fileLiveGtCoco" type="file" accept=".json" />
              <button id="btnEvalLiveGt" class="btn-primary btn-sm" type="button" disabled>导入标注并计算 IoU50 指标</button>
              <div id="liveGtStatus" style="font-size:11.5px;color:var(--muted);">未导入标注，仅展示模型预测框数（不代表板上实际器件数）。</div>
              <div id="liveGtResultBox" style="display:none;font-size:12px;"></div>
            </div>
          </details>
        </div>

        <details class="drawer">
          <summary>高级参数、预设切换与原始输出（按需展开）</summary>
          <div style="margin-top:8px;">
            <label class="fld">复合检测权重预设（默认 R15；A1 手动可选，已知局限：S16 曾误删 1 个真实 IC s16_front_yc0096）
              <select id="selCompositeDetectorPreset">
                <option value="fics_yolo11n_r15" selected>FICS-PCB 6类域适配 YOLO11n（R15 默认）</option>
                <option value="fics_yolo11n_r15_r19_a1_ic_reject">FICS-PCB R15 + R19 A1 IC 去误报（手动可选 · S16 已知局限）</option>
                <option value="wacv_yolo11n">WACV YOLO11n 旧基线</option>
              </select>
            </label>
            <label class="fld">复合检测模式
              <select id="selCompositeMode">
                <option value="COMPOSITE_YOLO" selected>COMPOSITE_YOLO</option>
                <option value="COMPOSITE_YOLO_QWEN3">COMPOSITE_YOLO_QWEN3</option>
              </select>
            </label>
            <label class="fld">纯 VLM 模型与运行预设（量化 / 适配器）
              <select id="selSingleModel">
                <option value="qwen3" selected>Qwen3-VL-2B-Instruct</option>
                <option value="qwen2">Qwen2-VL-2B-Instruct</option>
                <option value="deepseek">DeepSeek-VL-1.3B-chat</option>
              </select>
            </label>
            <div class="row">
              <select id="selModelMode" style="flex:1;">
                <option value="T" selected>T（切片推理 + NMS）</option>
                <option value="W">W（整图推理）</option>
              </select>
              <select id="livePreset" style="flex:1;">
                <option value="r5_qwen3_greedy_maxpx589824_tile768" selected>r5_qwen3_greedy_maxpx589824_tile768</option>
                <option value="r3_candidate">r3_candidate</option>
                <option value="r4_highres">r4_highres</option>
                <option value="r6_qwen3_tile768_b_other_strict">r6_qwen3_tile768_b_other_strict</option>
                <option value="r7_qwen3_tile768_dual_crop_consensus">r7_qwen3_tile768_dual_crop_consensus</option>
                <option value="r9_qwen3_nf4_base_tile768">r9_qwen3_nf4_base_tile768</option>
                <option value="r9_qwen3_nf4_adapter_tile768">r9_qwen3_nf4_adapter_tile768</option>
                <option value="r10_qwen3_bf16_tile768_evidence_prompt">r10_qwen3_bf16_tile768_evidence_prompt</option>
                <option value="baseline">baseline</option>
              </select>
            </div>
            <div class="row" style="margin-top:6px;">
              <button id="btnEntrySingleDefense" class="btn-sm" type="button">单模型主入口</button>
              <button id="btnEntryThreeModelCompare" class="btn-sm" type="button">三模型公平对比入口</button>
              <button id="btnStartCompositeJob" class="btn-sm" type="button" disabled>运行复合检测</button>
              <button id="btnStartSingleJob" class="btn-sm" type="button" disabled>运行单模型 VLM</button>
              <button id="btnStartCompare" class="btn-sm" type="button" style="display:none;">对比</button>
              <button id="btnExpCmpMd" class="btn-sm" type="button" style="display:none;">导出MD</button>
            </div>
            <div style="margin-top:8px;">
              <label class="fld">实验性空间指令对照
                <input id="inpExpSpatialInstruction" type="text" value="只找电阻，限图像左半边并计数" />
              </label>
              <button id="btnStartExpSpatialJob" class="btn-sm" type="button" disabled>启动隔离实验空间重推</button>
            </div>
            <div style="margin-top:8px;font-size:11px;color:var(--muted);">
              运行模式：1. 实时 GPU 推理 · 2. 已有六类结果筛选（CPU） · 3. 事后 COCO 标注评价 · 4. 历史只读回放<br/>
              事后 COCO GT 评价指标口径：目标类 GT (TP/FP/FN) · 同板全 6 类参考口径 · <code>phe.single_class_eval</code>
            </div>
            <pre id="rawOutputPre" style="margin-top:8px;background:#0f172a;color:#e2e8f0;padding:8px;border-radius:6px;font-size:11px;overflow:auto;max-height:180px;">暂无原始输出</pre>
          </div>
        </details>
      </div>
    </div>
  </section>

  <!-- Tab 2: 三模型对比 -->
  <section id="panelCompare" class="panel-hidden" role="tabpanel">
    <div class="card">
      <div class="card-title">
        <span>三模型同图串行对比（Qwen3 / Qwen2 / DeepSeek）</span>
        <span id="cmpUploadInfo" style="font-size:12px;color:var(--muted);">请先在「检测」页上传 PCB 图片</span>
      </div>
      <div class="row">
        <select id="selCmpMode">
          <option value="W" selected>共同配置：W（整图推理）</option>
          <option value="T">T（切片推理 + NMS）</option>
        </select>
        <select id="selCmpPreset">
          <option value="baseline" selected>共同基准（baseline）</option>
        </select>
        <select id="selCmpResultView" aria-label="对比结果视图">
          <option value="live_common" selected>当前实跑：共同配置结果</option>
          <option value="r28_tuned_readonly">只读参考：R28 各自调优锁盘对照（0 GPU）</option>
        </select>
        <button id="btnRunThreeCompare" class="btn-primary" type="button" disabled>开始三模型串行对比</button>
        <button id="btnExportCmpJson" type="button" disabled>导出对比数据</button>
      </div>
      <div id="cmpStatusNote" style="font-size:12px;color:var(--muted);margin-top:6px;">等待启动三模型对比。</div>
      <div class="viewport-bar" style="margin-top:8px;margin-bottom:0;">
        <span>联动看框：</span>
        <button id="btnCmpZoomIn" class="btn-sm" type="button">+</button>
        <button id="btnCmpZoomOut" class="btn-sm" type="button">-</button>
        <span id="cmpZoomPctLbl" style="min-width:42px;text-align:center;color:var(--muted);">100%</span>
        <button id="btnCmpZoomFit" class="btn-sm" type="button">适应窗口</button>
        <button id="btnCmpToggleBoxes" class="btn-sm" type="button">隐藏框</button>
        <select id="selCmpCoordView" aria-label="对比坐标基准">
          <option value="orig" selected>原图坐标</option>
          <option value="roi">ROI 裁切视图</option>
        </select>
      </div>
      <details class="drawer" style="margin-top:8px;">
        <summary>事后导入 COCO GT 标注评分（可选 · 0 新增 GPU）</summary>
        <div class="row" style="margin-top:6px;">
          <input id="fileCmpGtCoco" type="file" accept=".json" />
          <button id="btnEvalCmpGt" class="btn-primary btn-sm" type="button" disabled>导入标注并计算 IoU50 指标</button>
          <span id="cmpGtEvalStatus" style="font-size:12px;color:var(--muted);">未导入标注，仅展示模型预测与运行状态。</span>
        </div>
      </details>
      <div class="cmp-mobile-tabs">
        <button type="button" class="btn-sm cmp-mob-btn active" data-m="qwen3">Qwen3</button>
        <button type="button" class="btn-sm cmp-mob-btn" data-m="qwen2">Qwen2</button>
        <button type="button" class="btn-sm cmp-mob-btn" data-m="deepseek">DeepSeek</button>
      </div>
    </div>
    <div id="cmpReadonlyView" class="card panel-hidden" style="margin-top:10px;"></div>
    <div id="cmpColumns" class="cmp-cols"></div>
  </section>

  <!-- Tab 3: 历史结果 -->
  <section id="panelResearch" class="panel-hidden" role="tabpanel">
    <div class="card" style="margin-bottom:12px;">
      <div class="card-title">
        <span>本地实时任务读回（重启可恢复 · 0 新增 GPU）</span>
      </div>
      <div class="row">
        <select id="selRecentJob" style="flex:1;min-width:220px;"><option value="">（载入本地单任务记录）</option></select>
        <button id="btnLoadRecentJob" class="btn-primary" type="button">载入单任务</button>
      </div>
      <div class="row" style="margin-top:8px;">
        <select id="selRecentCmp" style="flex:1;min-width:220px;"><option value="">（载入三模型对比记录）</option></select>
        <button id="btnLoadRecentCmp" type="button">载入对比</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>历史研究回放（8 板只读归档 · 0 新增 GPU）</span>
      </div>
      <div class="notice n-info" style="margin-top:0;margin-bottom:8px;">
        来源：R12/R13 冻结 8 板只读归档（含 s19/s16 等离线冻结评测结果；S10 独立盲测与 S12/S24 保留板未接入本标签页）。
      </div>
      <div class="row">
        <select id="selResearchDataset"></select>
        <select id="selResearchBoard"></select>
        <select id="selResearchModel">
          <option value="qwen3" selected>qwen3</option>
          <option value="qwen2">qwen2</option>
          <option value="deepseek">deepseek</option>
        </select>
        <select id="selResearchMode">
          <option value="T_B_other" selected>T_B_other</option>
          <option value="T">T</option>
          <option value="W">W</option>
        </select>
        <button id="btnLoadResearchBoard" class="btn-primary" type="button">读取只读归档</button>
      </div>
      <div id="researchSummary" style="margin-top:8px;font-size:12.5px;color:var(--ink);">请选择归档板卡后点击读取。</div>
      <div style="margin-top:8px;max-height:56vh;overflow:auto;background:#0f172a;border-radius:8px;text-align:center;">
        <img id="researchImg" alt="研究回放图像" style="max-width:100%;height:auto;" />
      </div>
    </div>
  </section>
</main>
<script>
// Race-safe session & polling kernel (shared with Node VM regression tests; zero DOM errors in browser)
let LIVE_UPLOAD = null;
let LIVE_JOB_ID = null;
let LIVE_RESULT = null;
let LIVE_COMPARISON_ID = null;
let LIVE_COMPARISON = null;
let LIVE_POLL_TIMER = null;
let LIVE_ACTIVE_SESSION = null;
let _LIVE_SESSION_SEQ = 0;
let _FOCUS_SEQ = 0;

function _el(id) { return document.getElementById(id); }

function syncActionButtonsForSession(sess) {
  const ro = Boolean(LIVE_UPLOAD && LIVE_UPLOAD.read_only_playback_only);
  const running = Boolean(sess && sess.kind && !sess.terminalLocked);
  const bCmp = _el('btnStartCompare');
  const bJob = _el('btnStartJob');
  const bCan = _el('btnCancelJob');
  if (bCmp) bCmp.disabled = !LIVE_UPLOAD || ro || running;
  if (bJob) bJob.disabled = !LIVE_UPLOAD || ro || running;
  if (bCan) bCan.disabled = !running;
}

function clearLiveResultsForNewUpload() {
  _LIVE_SESSION_SEQ += 1;
  if (LIVE_POLL_TIMER) { clearInterval(LIVE_POLL_TIMER); LIVE_POLL_TIMER = null; }
  LIVE_ACTIVE_SESSION = null;
  LIVE_JOB_ID = null;
  LIVE_RESULT = null;
  LIVE_COMPARISON_ID = null;
  LIVE_COMPARISON = null;
  syncActionButtonsForSession(null);
}

function updateProgressUi(st) {
  const b = _el('stBadge_' + (st.model_key || 'qwen2'));
  if (b) b.textContent = st.status || '';
}

function updateComparisonProgressUi(cmp) {
  const b = _el('cmpStatusNote');
  if (b) b.textContent = cmp.status || '';
}

async function focusComparisonSlot(jobId) {
  const fseq = ++_FOCUS_SEQ;
  const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/result`);
  const d = await r.json();
  if (fseq !== _FOCUS_SEQ) return;
  LIVE_JOB_ID = jobId;
  LIVE_RESULT = d;
}

function _startSession(kind, targetId) {
  _LIVE_SESSION_SEQ += 1;
  if (LIVE_POLL_TIMER) { clearInterval(LIVE_POLL_TIMER); LIVE_POLL_TIMER = null; }
  const sess = {
    seq: _LIVE_SESSION_SEQ,
    kind,
    targetId,
    timerId: null,
    inFlight: false,
    pollSeq: 0,
    terminalLocked: false,
    transientFails: 0,
  };
  LIVE_ACTIVE_SESSION = sess;
  const tid = setInterval(() => {
    if (LIVE_ACTIVE_SESSION !== sess || sess.inFlight || sess.terminalLocked) return;
    return kind === 'job' ? pollJob(sess, tid) : pollComparison(sess, tid);
  }, 1500);
  sess.timerId = tid;
  LIVE_POLL_TIMER = tid;
  syncActionButtonsForSession(sess);
  return sess;
}

async function pollJob(sess, tid) {
  if (!sess || LIVE_ACTIVE_SESSION !== sess || sess.terminalLocked) return;
  const pseq = ++sess.pollSeq;
  sess.inFlight = true;
  try {
    const r = await fetch(`/api/jobs/${encodeURIComponent(sess.targetId)}`);
    const st = await r.json();
    if (LIVE_ACTIVE_SESSION !== sess || sess.terminalLocked || pseq !== sess.pollSeq) return;
    sess.transientFails = 0;
    const bar = _el('pollRecoveryBar');
    if (bar) bar.style.display = 'none';
    updateProgressUi(st);
    if (st.status === 'completed' || st.status === 'failed' || st.status === 'canceled') {
      sess.terminalLocked = true;
      if (LIVE_POLL_TIMER) { clearInterval(LIVE_POLL_TIMER); LIVE_POLL_TIMER = null; }
      syncActionButtonsForSession(sess);
      if (st.status === 'completed') {
        const rr = await fetch(`/api/jobs/${encodeURIComponent(sess.targetId)}/result`);
        if (LIVE_ACTIVE_SESSION === sess) LIVE_RESULT = await rr.json();
      }
    }
  } catch (e) {
    if (LIVE_ACTIVE_SESSION !== sess || sess.terminalLocked || pseq !== sess.pollSeq) return;
    sess.transientFails += 1;
    const bar = _el('pollRecoveryBar');
    if (bar) bar.style.display = 'block';
    if (sess.transientFails >= 3 && LIVE_POLL_TIMER) {
      clearInterval(LIVE_POLL_TIMER);
      LIVE_POLL_TIMER = null;
    }
  } finally {
    sess.inFlight = false;
  }
}

async function pollComparison(sess, tid) {
  if (!sess || LIVE_ACTIVE_SESSION !== sess || sess.terminalLocked) return;
  const pseq = ++sess.pollSeq;
  sess.inFlight = true;
  try {
    const r = await fetch(`/api/comparisons/${encodeURIComponent(sess.targetId)}`);
    const cmp = await r.json();
    if (LIVE_ACTIVE_SESSION !== sess || sess.terminalLocked || pseq !== sess.pollSeq) return;
    LIVE_COMPARISON = cmp;
    updateComparisonProgressUi(cmp);
    if (cmp.status === 'completed' || cmp.status === 'failed' || cmp.status === 'canceled') {
      sess.terminalLocked = true;
      if (LIVE_POLL_TIMER) { clearInterval(LIVE_POLL_TIMER); LIVE_POLL_TIMER = null; }
      syncActionButtonsForSession(sess);
    }
  } finally {
    sess.inFlight = false;
  }
}

(function initLegacyVmHooks() {
  const startCmpBtn = _el('btnStartCompare');
  const startJobBtn = _el('btnStartJob');
  const loadCmpBtn = _el('btnLoadRecentCmp');
  const loadJobBtn = _el('btnLoadRecentJob');
  const resumeBtn = _el('btnResumePoll');
  const expMdBtn = _el('btnExpCmpMd');
  if (startCmpBtn) {
    startCmpBtn.onclick = async () => {
      if (!LIVE_UPLOAD) return;
      const resp = await fetch('/api/comparisons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ upload_id: LIVE_UPLOAD.upload_id, model_keys: ['qwen2'], mode: 'W' }),
      });
      const cmp = await resp.json();
      LIVE_COMPARISON_ID = cmp.comparison_id;
      LIVE_COMPARISON = cmp;
      LIVE_RESULT = null;
      _startSession('comparison', cmp.comparison_id);
    };
  }
  if (startJobBtn) {
    startJobBtn.onclick = async () => {
      if (!LIVE_UPLOAD) return;
      const resp = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ upload_id: LIVE_UPLOAD.upload_id, model_key: 'qwen2', mode: 'W' }),
      });
      const st = await resp.json();
      LIVE_JOB_ID = st.job_id;
      LIVE_COMPARISON_ID = null;
      LIVE_COMPARISON = null;
      _startSession('job', st.job_id);
    };
  }
  if (loadCmpBtn) {
    loadCmpBtn.onclick = async () => {
      const sel = _el('selRecentCmp');
      const cid = sel ? sel.value : '';
      if (!cid) return;
      const r = await fetch(`/api/comparisons/${encodeURIComponent(cid)}`);
      const d = await r.json();
      LIVE_COMPARISON_ID = cid;
      LIVE_COMPARISON = d;
      LIVE_RESULT = null;
      if (d.status !== 'completed' && d.status !== 'failed' && d.status !== 'canceled') {
        _startSession('comparison', cid);
      }
    };
  }
  if (loadJobBtn) {
    loadJobBtn.onclick = async () => {
      const sel = _el('selRecentJob');
      const jid = sel ? sel.value : '';
      if (!jid) return;
      LIVE_JOB_ID = jid;
      const sess = _startSession('job', jid);
      await pollJob(sess, sess.timerId);
    };
  }
  if (resumeBtn) {
    resumeBtn.onclick = () => {
      if (LIVE_ACTIVE_SESSION && !LIVE_ACTIVE_SESSION.terminalLocked && !LIVE_POLL_TIMER) {
        const sess = LIVE_ACTIVE_SESSION;
        const tid = setInterval(() => {
          if (LIVE_ACTIVE_SESSION !== sess || sess.inFlight || sess.terminalLocked) return;
          if (sess.kind === 'job') pollJob(sess, tid);
          else pollComparison(sess, tid);
        }, 1500);
        sess.timerId = tid;
        LIVE_POLL_TIMER = tid;
      }
    };
  }
  if (expMdBtn) {
    expMdBtn.onclick = () => {
      if (LIVE_COMPARISON_ID) window.location.href = `/api/comparisons/${encodeURIComponent(LIVE_COMPARISON_ID)}/export/markdown`;
    };
  }
})();
</script>
<script src="/static_r28/r28_app.js"></script>
</body>
</html>
"""


class FinalDemoHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, service: LiveDemoService, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_bytes(self, code: int, body: bytes, content_type: str, download_name: Optional[str] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        self._send_bytes(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_body(self, max_bytes: int = 256 * 1024 * 1024) -> bytes:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            raise DemoError("请求体为空。", status_code=400)
        if length > max_bytes:
            raise DemoError("请求体超出大小限制（最大 256 MB）。", status_code=413)
        return self.rfile.read(length)

    def _handle_exception(self, exc: Exception) -> None:
        if isinstance(exc, DemoError):
            payload: Dict[str, Any] = {"error": str(exc)}
            if exc.details:
                payload["details"] = exc.details
            self._send_json(exc.status_code, payload)
        else:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"服务器内部异常：{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        path = parsed.path
        try:
            if path == "/":
                self._send_bytes(HTTPStatus.OK, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path in ("/static_r28/r28_app.css", "/static_r28/r28_app.js"):
                fname = path.rsplit("/", 1)[-1]
                asset_p = pathlib.Path(__file__).resolve().parent / "static_r28" / fname
                if not asset_p.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Static asset not found: {fname}"})
                else:
                    ctype = "text/css; charset=utf-8" if fname.endswith(".css") else "application/javascript; charset=utf-8"
                    self._send_bytes(HTTPStatus.OK, asset_p.read_bytes(), ctype)
            elif path == "/favicon.ico":
                self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            elif path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "research_available": bool(self.service.research_store and self.service.research_store.available)})
            elif path == "/api/config":
                self._send_json(HTTPStatus.OK, self.service.get_ui_defaults())
            elif path.startswith("/api/uploads/"):
                uid = urllib.parse.unquote(path[len("/api/uploads/"):])
                self._send_json(HTTPStatus.OK, self.service.get_upload_summary(uid))
            elif path.startswith("/api/upload_preview/"):
                uid = urllib.parse.unquote(path[len("/api/upload_preview/"):])
                max_w = int((qs.get("max_width") or ["1600"])[0])
                crop = (qs.get("crop") or [None])[0]
                source = (qs.get("source") or ["roi"])[0]
                jpg = self.service.render_upload_roi_jpeg(upload_id=uid, max_width=max_w, crop=crop, source=source)
                self._send_bytes(HTTPStatus.OK, jpg, "image/jpeg")
            elif path == "/api/jobs":
                uid_filter = (qs.get("upload_id") or [None])[0]
                self._send_json(HTTPStatus.OK, {"jobs": self.service.list_jobs(upload_id=uid_filter)})
            elif path.startswith("/api/jobs/"):
                parts = [p for p in path[len("/api/jobs/"):].split("/") if p]
                if len(parts) == 1:
                    self._send_json(HTTPStatus.OK, self.service.get_job_status(urllib.parse.unquote(parts[0])))
                elif len(parts) == 2 and parts[1] == "result":
                    self._send_json(HTTPStatus.OK, self.service.get_job_result(urllib.parse.unquote(parts[0])))
                elif len(parts) == 3 and parts[1] == "export":
                    view = (qs.get("view") or ["kept"])[0]
                    blob, ctype, fname = self.service.export_job(
                        job_id=urllib.parse.unquote(parts[0]),
                        kind=urllib.parse.unquote(parts[2]),
                        view=view,
                    )
                    self._send_bytes(HTTPStatus.OK, blob, ctype, download_name=fname)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            elif path.startswith("/api/composite/query_overlay/"):
                parts = [p for p in path[len("/api/composite/query_overlay/"):].split("/") if p]
                if len(parts) == 2:
                    blob, ctype, fname = self.service.export_composite_query_overlay(
                        job_id=urllib.parse.unquote(parts[0]),
                        query_id=urllib.parse.unquote(parts[1]),
                    )
                    self._send_bytes(HTTPStatus.OK, blob, ctype, download_name=fname)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            elif path.startswith("/api/composite/query_export/"):
                parts = [p for p in path[len("/api/composite/query_export/"):].split("/") if p]
                if len(parts) == 2:
                    blob, ctype, fname = self.service.export_composite_query_json(
                        job_id=urllib.parse.unquote(parts[0]),
                        query_id=urllib.parse.unquote(parts[1]),
                    )
                    self._send_bytes(HTTPStatus.OK, blob, ctype, download_name=fname)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            elif path.startswith("/api/composite/query/"):
                parts = [p for p in path[len("/api/composite/query/"):].split("/") if p]
                if len(parts) == 2:
                    self._send_json(
                        HTTPStatus.OK,
                        self.service.get_composite_query(
                            job_id=urllib.parse.unquote(parts[0]),
                            query_id=urllib.parse.unquote(parts[1]),
                        ),
                    )
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            elif path == "/api/comparisons":
                uid_filter = (qs.get("upload_id") or [None])[0]
                self._send_json(HTTPStatus.OK, {"comparisons": self.service.list_comparisons(upload_id=uid_filter)})
            elif path.startswith("/api/comparisons/"):
                parts = [p for p in path[len("/api/comparisons/"):].split("/") if p]
                if len(parts) == 1:
                    self._send_json(HTTPStatus.OK, self.service.get_comparison_status(urllib.parse.unquote(parts[0])))
                elif len(parts) == 3 and parts[1] == "export":
                    blob, ctype, fname = self.service.export_comparison(
                        comparison_id=urllib.parse.unquote(parts[0]),
                        kind=urllib.parse.unquote(parts[2]),
                    )
                    self._send_bytes(HTTPStatus.OK, blob, ctype, download_name=fname)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            elif path == "/api/research/overview":
                if not self.service.research_store:
                    self._send_json(HTTPStatus.OK, {"available": False, "notice": "未加载 research results_index.json。", "datasets": {}})
                else:
                    self._send_json(HTTPStatus.OK, self.service.research_store.get_overview())
            elif path == "/api/research/board":
                if not self.service.research_store:
                    raise DemoError("未加载 research results_index.json。", status_code=404)
                self._send_json(
                    HTTPStatus.OK,
                    self.service.research_store.get_board_view(
                        dataset_key=(qs.get("dataset") or [""])[0],
                        board_id=(qs.get("board") or [""])[0],
                        model_key=(qs.get("model") or [""])[0],
                        mode=(qs.get("mode") or [""])[0],
                    ),
                )
            elif path == "/api/research/image":
                if not self.service.research_store:
                    raise DemoError("未加载 research results_index.json。", status_code=404)
                jpg = self.service.research_store.render_image_jpeg(
                    dataset_key=(qs.get("dataset") or [""])[0],
                    board_id=(qs.get("board") or [""])[0],
                    max_width=int((qs.get("max_width") or ["1600"])[0]),
                    crop=(qs.get("crop") or [None])[0],
                )
                self._send_bytes(HTTPStatus.OK, jpg, "image/jpeg")
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
        except Exception as exc:
            self._handle_exception(exc)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/upload":
                raw = self._read_body()
                raw_name = urllib.parse.unquote(self.headers.get("X-Filename") or "upload.png")
                out = self.service.register_upload(raw_bytes=raw, raw_filename=raw_name)
                self._send_json(HTTPStatus.OK, out)
            elif path in ("/api/jobs", "/api/jobs/instruct", "/api/experimental/spatial_job"):
                body = json.loads(self._read_body().decode("utf-8"))
                u_inst = body.get("user_instruction")
                if path in ("/api/jobs/instruct", "/api/experimental/spatial_job") and u_inst is None:
                    u_inst = ""
                out = self.service.submit_job(
                    upload_id=str(body.get("upload_id") or ""),
                    model_key=str(body.get("model_key") or ""),
                    mode=str(body.get("mode") or ""),
                    preset=body.get("preset"),
                    detector_preset=body.get("detector_preset"),
                    source_proposal_job_id=body.get("source_proposal_job_id"),
                    user_instruction=u_inst,
                    allow_experimental_spatial=(path == "/api/experimental/spatial_job"),
                )
                self._send_json(HTTPStatus.ACCEPTED, out)
            elif path == "/api/jobs/verify_candidates":
                body = json.loads(self._read_body().decode("utf-8"))
                out = self.service.submit_verifier_only_job(
                    upload_id=str(body.get("upload_id") or ""),
                    source_predictions=body.get("predictions"),
                    preset=str(body.get("preset") or "r6_qwen3_tile768_b_other_strict"),
                    source_provenance=body.get("source_provenance"),
                    reuse_stage1=body.get("reuse_stage1"),
                )
                self._send_json(HTTPStatus.ACCEPTED, out)
            elif path == "/api/composite/query":
                body = json.loads(self._read_body().decode("utf-8"))
                out = self.service.execute_composite_query(
                    job_id=str(body.get("job_id") or ""),
                    query_text=str(body.get("query_text") or ""),
                )
                self._send_json(HTTPStatus.OK, out)
            elif path == "/api/composite/query_evaluate_gt":
                body = json.loads(self._read_body().decode("utf-8"))
                coco_gt = body.get("coco_gt")
                if not isinstance(coco_gt, dict):
                    raise DemoError("请求必须包含合法的 coco_gt JSON 对象。", status_code=400)
                tid = body.get("target_image_id")
                out = self.service.evaluate_composite_query_posthoc_gt(
                    job_id=str(body.get("job_id") or ""),
                    query_id=str(body.get("query_id") or ""),
                    coco_gt=coco_gt,
                    target_image_id=int(tid) if tid is not None else None,
                    confirm_manual_mapping=bool(body.get("confirm_manual_mapping")),
                    manual_mapping_reason=body.get("manual_mapping_reason"),
                )
                self._send_json(HTTPStatus.OK, out)
            elif path == "/api/comparisons":
                body = json.loads(self._read_body().decode("utf-8"))
                out = self.service.submit_comparison(
                    upload_id=str(body.get("upload_id") or ""),
                    model_keys=list(body.get("model_keys") or []),
                    mode=str(body.get("mode") or "W"),
                    preset=str(body.get("preset") or "baseline"),
                    include_qwen3_b_other=bool(body.get("include_qwen3_b_other")),
                )
                self._send_json(HTTPStatus.ACCEPTED, out)
            elif path == "/api/comparisons/from_jobs":
                body = json.loads(self._read_body().decode("utf-8"))
                out = self.service.create_comparison_from_jobs(
                    upload_id=str(body.get("upload_id") or ""),
                    mode=str(body.get("mode") or "W"),
                    job_ids=list(body.get("job_ids") or []),
                    addon_job_id=body.get("addon_job_id"),
                    addon_job_ids=body.get("addon_job_ids"),
                )
                self._send_json(HTTPStatus.OK, out)
            elif path.startswith("/api/comparisons/"):
                parts = [p for p in path[len("/api/comparisons/"):].split("/") if p]
                if len(parts) == 2 and parts[1] == "cancel":
                    out = self.service.cancel_comparison(urllib.parse.unquote(parts[0]))
                    self._send_json(HTTPStatus.OK, out)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            elif path.startswith("/api/jobs/"):
                parts = [p for p in path[len("/api/jobs/"):].split("/") if p]
                if len(parts) == 2 and parts[1] == "cancel":
                    out = self.service.cancel_job(urllib.parse.unquote(parts[0]))
                    self._send_json(HTTPStatus.OK, out)
                elif len(parts) == 2 and parts[1] == "evaluate_gt":
                    body = json.loads(self._read_body().decode("utf-8"))
                    coco_gt = body.get("coco_gt")
                    if not isinstance(coco_gt, dict):
                        raise DemoError("请求必须包含合法的 coco_gt JSON 对象。", status_code=400)
                    tid = body.get("target_image_id")
                    out = self.service.evaluate_job_with_coco(
                        job_id=urllib.parse.unquote(parts[0]),
                        coco_gt=coco_gt,
                        target_image_id=int(tid) if tid is not None else None,
                        confirm_manual_mapping=bool(body.get("confirm_manual_mapping")),
                        manual_mapping_reason=body.get("manual_mapping_reason"),
                    )
                    self._send_json(HTTPStatus.OK, out)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
        except Exception as exc:
            self._handle_exception(exc)


def _find_free_port(host: str, start_port: int, max_tries: int = 20) -> int:
    if start_port == 0:
        return 0
    for offset in range(max_tries):
        cand = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, cand))
                return cand
            except OSError:
                continue
    return 0


def create_demo_server(
    *,
    workspace_root: pathlib.Path,
    runtime_config_path: Optional[pathlib.Path] = None,
    results_index_path: Optional[pathlib.Path] = None,
    host: str = "127.0.0.1",
    port: int = 7862,
    allow_port_fallback: bool = True,
    service: Optional[LiveDemoService] = None,
) -> ThreadingHTTPServer:
    if service is None:
        service = LiveDemoService(
            workspace_root=workspace_root,
            runtime_config_path=runtime_config_path,
            research_store=None,
        )
        eff_idx = results_index_path
        if eff_idx is None:
            rt_idx = (service.runtime_config or {}).get("results_index_path") or (service.runtime_config or {}).get("results_index")
            if rt_idx:
                rt_p = pathlib.Path(str(rt_idx))
                if not rt_p.is_absolute() and runtime_config_path is not None:
                    rt_p = (pathlib.Path(runtime_config_path).resolve().parent / rt_p).resolve()
                eff_idx = rt_p
        service.research_store = ResearchReplayStore(eff_idx) if eff_idx else ResearchReplayStore(None)
    chosen_port = _find_free_port(host, int(port)) if allow_port_fallback else int(port)
    handler = functools.partial(FinalDemoHandler, service=service)
    srv = ThreadingHTTPServer((host, chosen_port), handler)
    srv.demo_service = service  # type: ignore[attr-defined]
    return srv


def resolve_server_launch_options(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse CLI arguments and optional JSON server startup --config with explicit CLI override precedence."""
    parser = argparse.ArgumentParser(description="FICS-PCB final GT-free demo & research replay server.")
    parser.add_argument("--config", help="Optional JSON demo server startup config (see configs/fics_final_demo.example.json). CLI flags override its values.")
    parser.add_argument("--workspace-root", help="Local directory for uploads and job outputs (required unless set via --config).")
    parser.add_argument("--runtime-config", help="Optional local model runtime YAML specifying local weights and qwen3 overlay.")
    parser.add_argument("--results-index", help="Optional results_index.json for read-only 8-board research replay.")
    parser.add_argument("--host", default=None, help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Listen port (default 7862).")
    parser.add_argument("--no-port-fallback", action="store_true", help="Disable fallback to next free port if busy.")
    parser.add_argument("--check-only", action="store_true", help="Validate config, local model/asset paths, and port availability without starting the server.")
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = {}
    cfg_base_dir: Optional[pathlib.Path] = None
    if args.config:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.is_file():
            raise SystemExit(f"Demo startup --config file not found: {cfg_path}")
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(
                f"Failed to parse --config as JSON ({cfg_path}): {exc}. "
                "Note: --config accepts JSON server startup settings, whereas model weights belong in --runtime-config YAML."
            ) from exc
        if not isinstance(loaded, dict):
            raise SystemExit(f"Demo startup --config must be a JSON object: {cfg_path}")
        if "models" in loaded:
            raise SystemExit(
                f"--config ({cfg_path}) appears to be a model runtime config with 'models'; pass it to --runtime-config instead."
            )
        cfg = loaded
        rel_anchor = str(cfg.get("relative_to") or "").strip().lower()
        if rel_anchor in ("config_dir", "config"):
            cfg_base_dir = cfg_path.resolve().parent
        elif rel_anchor in ("software_dir", "package_software_dir", "parent_dir"):
            cfg_base_dir = cfg_path.resolve().parent.parent
        elif rel_anchor in ("cwd", ""):
            cfg_base_dir = None
        else:
            raise SystemExit(
                f"Unsupported 'relative_to' value in --config ({cfg_path}): {cfg.get('relative_to')!r}. "
                "Supported values: 'config_dir', 'software_dir', or 'cwd'."
            )

    def _resolve_cfg_path(cli_val: Optional[str], cfg_val: Any) -> Optional[pathlib.Path]:
        if cli_val is not None:
            return pathlib.Path(str(cli_val))
        if not cfg_val:
            return None
        p = pathlib.Path(str(cfg_val))
        if not p.is_absolute() and cfg_base_dir is not None:
            return (cfg_base_dir / p).resolve()
        return p

    ws_path = _resolve_cfg_path(args.workspace_root, cfg.get("workspace_root"))
    if ws_path is None:
        raise SystemExit("Missing workspace_root: provide --workspace-root or set 'workspace_root' in --config JSON.")

    rt_path = _resolve_cfg_path(args.runtime_config, cfg.get("runtime_config"))
    idx_path = _resolve_cfg_path(args.results_index, cfg.get("results_index"))
    host = args.host if args.host is not None else str(cfg.get("host") or "127.0.0.1")
    port = args.port if args.port is not None else int(cfg.get("port") if cfg.get("port") is not None else 7862)
    if args.no_port_fallback:
        allow_fallback = False
    elif "allow_port_fallback" in cfg:
        allow_fallback = bool(cfg["allow_port_fallback"])
    else:
        allow_fallback = True

    return {
        "workspace_root": ws_path,
        "runtime_config_path": rt_path,
        "results_index_path": idx_path,
        "host": host,
        "port": port,
        "allow_port_fallback": allow_fallback,
        "check_only": bool(args.check_only),
    }


def preflight_validate_launch(opts: Dict[str, Any]) -> Dict[str, Any]:
    """Validate local runtime config, model/overlay/weight files, results index, and bind port before serving."""
    import os
    import socket

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "true")

    rt_path: Optional[pathlib.Path] = opts.get("runtime_config_path")
    checked_paths: Dict[str, str] = {}
    if rt_path is not None:
        if not rt_path.is_file():
            raise SystemExit(f"Runtime config file not found: {rt_path}")
        checked_paths["runtime_config"] = str(rt_path)
        try:
            import yaml  # type: ignore
            rt_data = yaml.safe_load(rt_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise SystemExit(f"Failed to parse runtime config YAML ({rt_path}): {exc}") from exc
        models_cfg = rt_data.get("models") if isinstance(rt_data, dict) else None
        if isinstance(models_cfg, dict):
            for model_key, entry in models_cfg.items():
                if not isinstance(entry, dict):
                    continue
                for path_field in ("local_dir", "model_dir", "overlay_dir", "dependency_overlay", "weights", "crop_verifier_weights"):
                    val = entry.get(path_field)
                    if not val:
                        continue
                    p = pathlib.Path(str(val))
                    if not p.is_absolute():
                        p = (rt_path.resolve().parent / p).resolve()
                    if not p.exists():
                        raise SystemExit(f"Missing local model asset for {model_key}.{path_field}: {p}")
                    checked_paths[f"{model_key}.{path_field}"] = str(p)

    idx_path: Optional[pathlib.Path] = opts.get("results_index_path")
    if idx_path is not None:
        if not idx_path.is_file():
            raise SystemExit(f"Results index file not found: {idx_path}")
        checked_paths["results_index"] = str(idx_path)

    host = str(opts["host"])
    port = int(opts["port"])
    if not bool(opts.get("allow_port_fallback", True)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError as exc:
                raise SystemExit(
                    f"Port {host}:{port} is already in use and allow_port_fallback is disabled ({exc}). "
                    "Stop the conflicting process or pass --port <free_port>."
                ) from exc

    return {
        "ok": True,
        "host": host,
        "port": port,
        "allow_port_fallback": bool(opts.get("allow_port_fallback", True)),
        "workspace_root": str(opts["workspace_root"]),
        "checked_paths": checked_paths,
        "offline_env": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
            "YOLO_OFFLINE": os.environ.get("YOLO_OFFLINE", ""),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    opts = resolve_server_launch_options(argv)
    preflight = preflight_validate_launch(opts)
    if opts.get("check_only"):
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        return 0
    try:
        srv = create_demo_server(
            workspace_root=opts["workspace_root"],
            runtime_config_path=opts["runtime_config_path"],
            results_index_path=opts["results_index_path"],
            host=opts["host"],
            port=opts["port"],
            allow_port_fallback=opts["allow_port_fallback"],
        )
    except OSError as exc:
        raise SystemExit(
            f"Failed to bind http://{opts['host']}:{opts['port']}/ (allow_port_fallback={opts['allow_port_fallback']}): {exc}"
        ) from exc
    print(f"FICS Final Demo listening on http://{opts['host']}:{srv.server_address[1]}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.demo_service.shutdown()  # type: ignore[attr-defined]
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
