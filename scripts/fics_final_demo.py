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
<title>基于开源视觉语言大模型的电子元器件检测系统</title>
<style>
  :root { --bg: #0b1220; --panel: #131e33; --panel-2: #0f172a; --line: #27364f; --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --danger: #ef4444; --purple: #c084fc; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Microsoft YaHei", system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); }
  .header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 20px; background: #10203d; border-bottom: 1px solid #1d4ed8; }
  .header h1 { margin: 0; font-size: 17px; font-weight: 700; color: #e0f2fe; }
  .header .sub { font-size: 12px; color: #93c5fd; margin-top: 2px; }
  .tabs { display: flex; gap: 8px; }
  .tab-btn { background: var(--panel-2); color: var(--muted); border: 1px solid var(--line); border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 600; cursor: pointer; }
  .tab-btn.active { background: #0c4a6e; color: #fff; border-color: var(--accent); }
  .wrap { padding: 14px 20px; max-width: 1680px; margin: 0 auto; }
  .grid-main { display: grid; grid-template-columns: minmax(340px, 430px) minmax(0, 1fr); gap: 14px; align-items: start; }
  @media (max-width: 1100px) { .grid-main { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin-bottom: 12px; }
  .card h2 { margin: 0 0 10px 0; font-size: 14px; color: #e2e8f0; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  label.fld { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); width: 100%; margin-bottom: 8px; }
  select, input[type="file"], input[type="number"], button { background: var(--panel-2); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; font-size: 13px; }
  button { cursor: pointer; font-weight: 600; }
  button.primary { background: #0284c7; border-color: #38bdf8; color: #fff; }
  button.primary:hover:not(:disabled) { background: #0369a1; }
  button.danger { background: #7f1d1d; border-color: #ef4444; color: #fee2e2; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .notice { padding: 10px 12px; border-radius: 8px; font-size: 12px; line-height: 1.5; margin-bottom: 10px; }
  .n-info { background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.35); color: #bae6fd; }
  .n-warn { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); color: #fde68a; }
  .n-err { background: rgba(239,68,68,0.14); border: 1px solid rgba(239,68,68,0.5); color: #fecaca; }
  .n-ok { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.4); color: #bbf7d0; }
  .progress-track { width: 100%; height: 10px; background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px; overflow: hidden; margin: 8px 0; }
  .progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #0ea5e9, #22c55e); transition: width 0.25s ease; }
  .kpis { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
  .kpi { background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .kpi .k { font-size: 11px; color: var(--muted); }
  .kpi .v { font-size: 16px; font-weight: 700; margin-top: 2px; }
  .canvas-wrap { position: relative; width: 100%; overflow: auto; max-height: 68vh; background: #020617; border: 1px solid var(--line); border-radius: 8px; }
  .stage { position: relative; display: inline-block; transform-origin: top left; }
  .stage img { display: block; max-width: none; }
  .stage svg { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }
  .tbl-wrap { max-height: 240px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 6px 8px; border-bottom: 1px solid #1e293b; text-align: left; }
  th { background: #0f172a; color: #cbd5e1; position: sticky; top: 0; }
  tr.clickable:hover { background: #1e293b; cursor: pointer; }
  .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .b-keep { background: rgba(56,189,248,0.2); color: #7dd3fc; }
  .b-rej { background: rgba(244,63,94,0.2); color: #fda4af; }
  .b-tp { background: rgba(56,189,248,0.2); color: #7dd3fc; }
  .b-fp { background: rgba(239,68,68,0.2); color: #fca5a5; }
  .b-fn { background: rgba(245,158,11,0.2); color: #fde047; }
  pre.raw-pre { white-space: pre-wrap; word-break: break-word; max-height: 170px; overflow: auto; background: #020617; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-size: 11.5px; margin: 6px 0 0 0; }
  .research-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; }
  @media (max-width: 1050px) { .research-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>基于开源视觉语言大模型的电子元器件检测系统</h1>
      <div class="sub">FICS-PCB 6 类目标 · 无 GT 新板实时检测 / 8 板研究数据严格隔离回放</div>
    </div>
    <div class="tabs">
      <button id="tabBtnLive" class="tab-btn active" type="button">新图无 GT 整板检测</button>
      <button id="tabBtnResearch" class="tab-btn" type="button">8 板研究结果回放（只读）</button>
    </div>
  </div>

  <!-- TAB 1: LIVE GT-FREE WORKFLOW -->
  <div id="tabLive" class="wrap">
    <div class="grid-main">
      <div>
        <div class="card">
          <h2>1. 上传新 PCB 图像（无需 Ground Truth）</h2>
          <div class="notice n-info">
            新板检测全流程不依赖、不读取任何 COCO 真值标注；系统自动完成白边 ROI 裁切（阈值 &lt;248 + 32px）与 1024px 切片预检。
          </div>
          <label class="fld">选择 PCB 图像文件（PNG / JPG / TIFF）
            <input id="fileBoard" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp" />
          </label>
          <div class="row">
            <button id="btnUpload" class="primary" type="button">上传并预检 ROI / 切片</button>
          </div>
          <div id="uploadMsg" class="notice n-warn" style="display:none;margin-top:8px;"></div>
          <div id="uploadMetaBox" style="margin-top:10px;font-size:12px;color:var(--muted);line-height:1.55;"></div>
        </div>

        <div class="card">
          <h2>2. 模型与推理模式选择</h2>
          <label class="fld">视觉语言模型
            <select id="liveModel">
              <option value="qwen3">Qwen3-VL-2B-Instruct（主方法基座）</option>
              <option value="qwen2">Qwen2-VL-2B-Instruct</option>
              <option value="deepseek">deepseek-vl-1.3b-chat</option>
            </select>
          </label>
          <label class="fld">检测模式
            <select id="liveMode">
              <option value="T_B_other">切片+背景核验（冻结方法 T + B_other · 仅 Qwen3）</option>
              <option value="T">T（1024px 切片 + 0.50 NMS）</option>
              <option value="W">W（整 ROI 推理 · 无切片 NMS）</option>
            </select>
          </label>
          <div id="modeHint" style="font-size:11.5px;color:var(--muted);margin-bottom:8px;">
            当前采用预注册冻结方法 Qwen3_T_plus_B_other：先完成全板 T 切片与去重，再对每个候选框做 10% 外扩裁剪与二阶段过滤。
          </div>
          <div class="row">
            <button id="btnStartJob" class="primary" type="button" disabled>开始整板检测</button>
            <button id="btnCancelJob" class="danger" type="button" disabled>安全取消当前任务</button>
          </div>
          <div class="row" style="margin-top:8px;">
            <select id="selRecentJob" style="flex:1;min-width:180px;"><option value="">（可从本地工作区重新载入近期任务）</option></select>
            <button id="btnLoadRecentJob" type="button">载入</button>
          </div>
          <div id="jobErrorBox" class="notice n-err" style="display:none;margin-top:8px;"></div>
          <div id="jobProgressCard" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid var(--line);">
            <div style="display:flex;justify-content:space-between;font-size:12px;">
              <strong id="jobStageLbl">-</strong>
              <span id="jobTimeLbl" style="color:var(--muted);">-</span>
            </div>
            <div class="progress-track"><div id="jobProgressFill" class="progress-fill"></div></div>
            <div id="jobCountsLbl" style="font-size:12px;color:var(--muted);line-height:1.5;"></div>
          </div>
        </div>

        <div class="card">
          <h2>3. 可选：事后上传 COCO GT 对齐评价（默认未评价）</h2>
          <div id="gtStatusBanner" class="notice n-warn">未提供真值：当前仅输出 GT-free 整板预测与类别统计。</div>
          <label class="fld">事后评价 COCO JSON 文件
            <input id="fileGtCoco" type="file" accept=".json" />
          </label>
          <label class="fld">多图 COCO 时的目标 image_id（单图留空自动核对文件名与尺寸）
            <input id="inpGtImageId" type="number" placeholder="自动匹配（留空）" />
          </label>
          <label style="display:flex;gap:6px;align-items:flex-start;font-size:12px;color:var(--muted);margin-bottom:6px;">
            <input id="chkGtManualConfirm" type="checkbox" />
            <span>仅用于人为改名图像：我已人工确认上传图与所选 COCO GT 为同一板卡（仍严格拒绝 SHA-256 冲突，且不标记为字节已验证）</span>
          </label>
          <label class="fld">人工映射说明（可选）
            <input id="inpGtManualReason" type="text" placeholder="例如：原图由 s20_front.tif 重命名为现场编号" />
          </label>
          <button id="btnEvalGt" type="button" disabled>提交事后 GT 对齐评价</button>
          <div id="gtEvalMsg" class="notice n-err" style="display:none;margin-top:8px;"></div>
        </div>
      </div>

      <div>
        <div class="card">
          <div class="row" style="justify-content:space-between;margin-bottom:8px;">
            <strong id="liveCanvasTitle" style="font-size:14px;">整板可视化窗口（请先上传图像并启动推理）</strong>
            <div class="row">
              <select id="liveOverlayView">
                <option value="kept">核验后保留 / 最终结果</option>
                <option value="pre_verifier">原 T 候选（核验前）</option>
                <option value="with_rejected">叠加显示被 B_other 过滤候选</option>
              </select>
              <select id="liveZoom">
                <option value="fit">自适应视野</option>
                <option value="1">1.0x</option>
                <option value="2">2.0x</option>
              </select>
              <button id="btnLiveResetCrop" type="button">显示全 ROI</button>
            </div>
          </div>
          <div id="liveCropStatus" style="font-size:12px;color:var(--muted);margin-bottom:6px;">等待图像…</div>
          <div id="liveCanvasWrap" class="canvas-wrap">
            <div id="liveStage" class="stage">
              <img id="liveImg" alt="Live PCB ROI" />
              <svg id="liveSvg" preserveAspectRatio="none"></svg>
            </div>
          </div>
        </div>

        <div class="card" id="liveResultCard" style="display:none;">
          <div class="row" style="justify-content:space-between;margin-bottom:10px;">
            <h2 style="margin:0;border:none;padding:0;">检测结果汇总与导出</h2>
            <div class="row">
              <button id="btnExpJson" type="button">下载标准 JSON</button>
              <button id="btnExpPng" type="button">下载标注图 PNG</button>
              <button id="btnExpMd" type="button">下载摘要 (.md)</button>
            </div>
          </div>
          <div id="liveKpis" class="kpis"></div>
          <div id="liveNote" style="font-size:12px;color:var(--muted);margin-bottom:8px;"></div>
          <div class="row" style="gap:12px;align-items:flex-start;">
            <div style="flex:1;min-width:250px;">
              <div style="font-size:12px;font-weight:700;margin-bottom:4px;">6 类检出计数</div>
              <div class="tbl-wrap" style="max-height:190px;">
                <table><thead><tr><th>类别</th><th>最终保留数</th><th>原 T 候选数</th></tr></thead><tbody id="liveClassBody"></tbody></table>
              </div>
            </div>
            <div style="flex:1.5;min-width:300px;">
              <div style="font-size:12px;font-weight:700;margin-bottom:4px;">预测目标列表（点击行可局部放大定位）</div>
              <div class="tbl-wrap" style="max-height:190px;">
                <table><thead><tr><th>状态 / 核验标签</th><th>类别</th><th>ROI 坐标 [x1,y1,x2,y2]</th></tr></thead><tbody id="liveItemsBody"></tbody></table>
              </div>
            </div>
          </div>
          <div id="gtPerClassWrap" style="display:none;margin-top:10px;">
            <div style="font-size:12px;font-weight:700;margin-bottom:4px;">事后 GT 各类别指标（GT=0 时 Recall/F1 标为 N/A (GT=0)）</div>
            <div class="tbl-wrap" style="max-height:170px;">
              <table><thead><tr><th>类别</th><th>GT</th><th>Pred</th><th>TP/FP/FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody id="liveGtPerClassBody"></tbody></table>
            </div>
          </div>
          <details style="margin-top:10px;">
            <summary style="font-size:12px;color:var(--muted);cursor:pointer;">展开技术详情（原始调用输出 / token / 耗时与校验信息）</summary>
            <div style="margin-top:8px;">
              <div class="row" style="justify-content:space-between;">
                <span style="font-size:12px;color:var(--muted);">原始调用输出查看（不含硬编码置信度）</span>
                <select id="selLiveCall" style="max-width:340px;"></select>
              </div>
              <div id="liveCallMeta" style="font-size:11.5px;color:var(--muted);margin-top:4px;"></div>
              <pre id="liveCallRaw" class="raw-pre">-</pre>
            </div>
          </details>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: 8-BOARD RESEARCH REPLAY -->
  <div id="tabResearch" class="wrap" style="display:none;">
    <div id="researchUnavailable" class="notice n-warn" style="display:none;"></div>
    <div id="researchContent" style="display:none;">
      <div class="research-grid">
        <div class="card">
          <h2>历史 / 开发对照参考（6 板 · 268 GT · 非最终盲测成绩）</h2>
          <div id="histSummaryBox" style="font-size:12px;color:var(--muted);line-height:1.55;"></div>
        </div>
        <div class="card">
          <h2>最终盲测保留集（2 板 s19 + s16 · 107 GT · 冻结方法单次盲评）</h2>
          <div id="heldoutSummaryBox" style="font-size:12px;color:var(--muted);line-height:1.55;"></div>
        </div>
      </div>

      <div class="card">
        <div class="row" style="gap:10px;margin-bottom:10px;">
          <label class="fld" style="width:auto;margin:0;">数据集范围（严格隔离）
            <select id="resDataset"></select>
          </label>
          <label class="fld" style="width:auto;margin:0;">板卡
            <select id="resBoard"></select>
          </label>
          <label class="fld" style="width:auto;margin:0;">模型
            <select id="resModel">
              <option value="qwen3">Qwen3-VL-2B-Instruct</option>
              <option value="qwen2">Qwen2-VL-2B-Instruct</option>
              <option value="deepseek">deepseek-vl-1.3b-chat</option>
            </select>
          </label>
          <label class="fld" style="width:auto;margin:0;">模式
            <select id="resMode">
              <option value="W">W（整 ROI）</option>
              <option value="T">T（切片 + NMS）</option>
              <option value="T_B_other">T + B_other（仅 Held-Out Qwen3）</option>
            </select>
          </label>
          <label class="fld" style="width:auto;margin:0;">缩放
            <select id="resZoom">
              <option value="fit">自适应</option>
              <option value="1">1.0x</option>
              <option value="2">2.0x</option>
            </select>
          </label>
          <button id="btnResResetCrop" type="button" style="align-self:flex-end;">显示全 ROI</button>
          <div class="row" style="margin-left:auto;font-size:12px;">
            <label><input type="checkbox" id="chkResTP" checked /> <span class="badge b-tp">TP</span></label>
            <label><input type="checkbox" id="chkResFP" checked /> <span class="badge b-fp">FP</span></label>
            <label><input type="checkbox" id="chkResFN" checked /> <span class="badge b-fn">FN</span></label>
            <label id="lblResPreT" style="display:none;"><input type="checkbox" id="chkResPreT" /> 显示 T 核验前被剔除框</label>
          </div>
        </div>
        <div id="resErr" class="notice n-err" style="display:none;"></div>
        <div class="grid-main" style="grid-template-columns:minmax(0,1.75fr) minmax(320px,1fr);">
          <div>
            <div id="resBoardTitle" style="font-size:13px;font-weight:700;margin-bottom:6px;">-</div>
            <div id="resCanvasWrap" class="canvas-wrap">
              <div id="resStage" class="stage">
                <img id="resImg" alt="Research PCB ROI" />
                <svg id="resSvg" preserveAspectRatio="none"></svg>
              </div>
            </div>
            <div class="row" style="justify-content:space-between;margin-top:8px;">
              <span style="font-size:12px;color:var(--muted);">原始调用选择器</span>
              <select id="selResCall" style="max-width:340px;"></select>
            </div>
            <div id="resCallMeta" style="font-size:11.5px;color:var(--muted);margin-top:4px;"></div>
            <pre id="resCallRaw" class="raw-pre">-</pre>
          </div>
          <div>
            <div id="resKpis" class="kpis"></div>
            <div id="resVerifierNote" style="font-size:12px;color:var(--muted);margin-bottom:8px;"></div>
            <div class="tbl-wrap" style="max-height:250px;">
              <table><thead><tr><th>状态</th><th>类别</th><th>IoU / ROI坐标</th></tr></thead><tbody id="resItemsBody"></tbody></table>
            </div>
            <details style="margin-top:10px;">
              <summary style="font-size:12px;font-weight:700;color:var(--muted);cursor:pointer;">展开技术详情：文件来源与 SHA-256 校验溯源</summary>
              <pre id="resProvPre" class="raw-pre" style="max-height:150px;"></pre>
            </details>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const CLASSES = ["capacitor", "resistor", "inductor", "transistor", "diode", "ic"];
let LIVE_UPLOAD = null;
let LIVE_JOB_ID = null;
let LIVE_RESULT = null;
let LIVE_CROP = null;
let LIVE_POLL_TIMER = null;

let RES_OVERVIEW = null;
let RES_CURRENT = null;
let RES_CROP = null;

async function apiJson(url, opts) {
  const r = await fetch(url, opts);
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
  return d;
}
function pct(v) { return (v === null || v === undefined) ? "N/A" : (v * 100).toFixed(2) + "%"; }

// Tabs
document.getElementById("tabBtnLive").onclick = () => {
  document.getElementById("tabBtnLive").classList.add("active");
  document.getElementById("tabBtnResearch").classList.remove("active");
  document.getElementById("tabLive").style.display = "block";
  document.getElementById("tabResearch").style.display = "none";
};
document.getElementById("tabBtnResearch").onclick = () => {
  document.getElementById("tabBtnResearch").classList.add("active");
  document.getElementById("tabBtnLive").classList.remove("active");
  document.getElementById("tabLive").style.display = "none";
  document.getElementById("tabResearch").style.display = "block";
  if (!RES_OVERVIEW) loadResearchOverview();
};

// Model / mode lock
function syncModeOptions() {
  const mk = document.getElementById("liveModel").value;
  const selMode = document.getElementById("liveMode");
  const bOpt = selMode.querySelector('option[value="T_B_other"]');
  if (mk !== "qwen3") {
    bOpt.disabled = true;
    if (selMode.value === "T_B_other") selMode.value = "T";
    document.getElementById("modeHint").textContent = "注意：冻结二次核验方法 T + B_other 仅适用于 Qwen3-VL-2B-Instruct；当前模型支持 W 与 T。";
  } else {
    bOpt.disabled = false;
    document.getElementById("modeHint").textContent = selMode.value === "T_B_other"
      ? "当前采用预注册冻结方法 Qwen3_T_plus_B_other：先完成全板 T 切片与 NMS，再对每个候选框做 10% 外扩裁剪与二阶段过滤。"
      : "Qwen3 兼容 W（整 ROI）、T（切片+NMS）以及冻结方法 T + B_other。";
  }
}
document.getElementById("liveModel").onchange = syncModeOptions;
document.getElementById("liveMode").onchange = syncModeOptions;

// Step 1: Upload
document.getElementById("btnUpload").onclick = async () => {
  const fileInput = document.getElementById("fileBoard");
  const msgEl = document.getElementById("uploadMsg");
  msgEl.style.display = "none";
  if (!fileInput.files || !fileInput.files.length) {
    msgEl.className = "notice n-err";
    msgEl.textContent = "请先选择一张 PCB 图像文件。";
    msgEl.style.display = "block";
    return;
  }
  const f = fileInput.files[0];
  document.getElementById("btnUpload").disabled = true;
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(f.name) },
      body: f,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
    LIVE_UPLOAD = data;
    LIVE_RESULT = null;
    LIVE_CROP = null;
    document.getElementById("btnStartJob").disabled = false;
    document.getElementById("btnEvalGt").disabled = true;
    document.getElementById("liveResultCard").style.display = "none";
    document.getElementById("gtStatusBanner").textContent = "未提供真值：当前仅输出 GT-free 整板预测与类别统计。";
    document.getElementById("uploadMetaBox").innerHTML =
      `<div>已通过 GT-free 预检：<strong>${data.safe_filename}</strong></div>` +
      `<div>原图尺寸：${data.orig_wh[0]} x ${data.orig_wh[1]} | ROI：${data.roi_width} x ${data.roi_height} | 切片数：<strong>${data.tile_count}</strong></div>` +
      `<details style="margin-top:4px;"><summary style="cursor:pointer;">展开技术校验详情 (upload_id / ROI 偏移 / SHA-256)</summary>` +
      `<div style="margin-top:4px;">upload_id: <code>${data.upload_id}</code> | ROI offset: [${data.roi_offset_x}, ${data.roi_offset_y}]<br/>SHA-256: <code>${data.source_image_sha256}</code></div></details>`;
    renderLiveCanvas();
  } catch (err) {
    msgEl.className = "notice n-err";
    msgEl.textContent = "上传失败：" + err.message;
    msgEl.style.display = "block";
  } finally {
    document.getElementById("btnUpload").disabled = false;
  }
};

// Step 2: Start / Cancel Job
document.getElementById("btnStartJob").onclick = async () => {
  if (!LIVE_UPLOAD) return;
  const errBox = document.getElementById("jobErrorBox");
  errBox.style.display = "none";
  const mk = document.getElementById("liveModel").value;
  const mode = document.getElementById("liveMode").value;
  try {
    const job = await apiJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: LIVE_UPLOAD.upload_id, model_key: mk, mode: mode }),
    });
    LIVE_JOB_ID = job.job_id;
    LIVE_RESULT = null;
    document.getElementById("liveResultCard").style.display = "none";
    document.getElementById("btnStartJob").disabled = true;
    document.getElementById("btnCancelJob").disabled = false;
    document.getElementById("jobProgressCard").style.display = "block";
    updateProgressUi(job);
    if (LIVE_POLL_TIMER) clearInterval(LIVE_POLL_TIMER);
    LIVE_POLL_TIMER = setInterval(pollJob, 800);
  } catch (err) {
    errBox.textContent = err.message;
    errBox.style.display = "block";
  }
};

document.getElementById("btnCancelJob").onclick = async () => {
  if (!LIVE_JOB_ID) return;
  document.getElementById("btnCancelJob").disabled = true;
  try {
    const st = await apiJson(`/api/jobs/${encodeURIComponent(LIVE_JOB_ID)}/cancel`, { method: "POST" });
    updateProgressUi(st);
  } catch (err) {
    document.getElementById("jobErrorBox").textContent = "取消失败：" + err.message;
    document.getElementById("jobErrorBox").style.display = "block";
  }
};

function updateProgressUi(st) {
  document.getElementById("jobStageLbl").textContent = `[${st.status}] ${st.stage_label}`;
  const etaTxt = (st.eta_seconds !== null && st.eta_seconds !== undefined) ? ` | 预计剩余 ${st.eta_seconds}s` : "";
  document.getElementById("jobTimeLbl").textContent = `已耗时 ${st.elapsed_seconds}s${etaTxt}`;
  const planTotal = (st.base_calls_planned || 0) + (st.verifier_calls_planned || 0);
  const doneTotal = (st.base_calls_completed || 0) + (st.verifier_calls_completed || 0);
  const ratio = planTotal > 0 ? Math.min(100, Math.round((doneTotal / planTotal) * 100)) : 5;
  document.getElementById("jobProgressFill").style.width = (st.status === "completed" ? 100 : ratio) + "%";
  let detail = `基线调用进度：${st.base_calls_completed} / ${st.base_calls_planned}`;
  if (st.mode === "T_B_other") {
    detail += ` | 二次核验 (B_other) 裁剪进度：${st.verifier_calls_completed} / ${st.verifier_calls_planned}`;
  }
  if (st.gpu_gate) {
    detail += ` | VRAM 门禁：${st.gpu_gate.passed ? "通过" : "未通过"}`;
  }
  document.getElementById("jobCountsLbl").textContent = detail;
}

async function pollJob() {
  if (!LIVE_JOB_ID) return;
  try {
    const st = await apiJson(`/api/jobs/${encodeURIComponent(LIVE_JOB_ID)}`);
    updateProgressUi(st);
    if (["completed", "failed", "cancelled", "incomplete"].includes(st.status)) {
      clearInterval(LIVE_POLL_TIMER);
      LIVE_POLL_TIMER = null;
      document.getElementById("btnStartJob").disabled = false;
      document.getElementById("btnCancelJob").disabled = true;
      if (st.status === "completed") {
        LIVE_RESULT = await apiJson(`/api/jobs/${encodeURIComponent(LIVE_JOB_ID)}/result`);
        document.getElementById("btnEvalGt").disabled = false;
        renderLiveResult();
      } else {
        const eb = document.getElementById("jobErrorBox");
        eb.textContent = st.error_message || ("任务终止状态：" + st.status);
        eb.style.display = "block";
      }
    }
  } catch (err) {
    clearInterval(LIVE_POLL_TIMER);
    LIVE_POLL_TIMER = null;
    document.getElementById("btnStartJob").disabled = false;
    document.getElementById("btnCancelJob").disabled = true;
  }
}

function renderLiveResult() {
  if (!LIVE_RESULT) return;
  document.getElementById("liveResultCard").style.display = "block";
  const r = LIVE_RESULT;
  const ovSel = document.getElementById("liveOverlayView");
  const hasPre = Array.isArray(r.pre_verifier_candidates);
  ovSel.querySelector('option[value="pre_verifier"]').disabled = !hasPre;
  ovSel.querySelector('option[value="with_rejected"]').disabled = !hasPre;
  if (!hasPre && ovSel.value !== "kept") ovSel.value = "kept";

  const cards = [
    ["模型 / 模式", `${r.model_key} / ${r.mode}`],
    ["最终保留目标数", String(r.kept_predictions_count)],
    ["切片 Post-NMS 候选数", String(r.post_nms_candidates_count)],
  ];
  if (r.verifier_summary) {
    cards.push(["B_other 剔除 other", String(r.verifier_summary.rejected_as_other)]);
  }
  if (r.post_hoc_eval) {
    const phe = r.post_hoc_eval;
    const m = phe.metrics;
    const rDisp = m.recall_display || pct(m.recall);
    const f1Disp = m.f1_display || pct(m.f1);
    cards.push(["事后 GT (TP/FP/FN)", `${m.tp} / ${m.fp} / ${m.fn}`]);
    cards.push(["Precision / Recall / F1", `${pct(m.precision)} / ${rDisp} / ${f1Disp}`]);
    document.getElementById("gtStatusBanner").className = phe.image_identity_verified ? "notice n-ok" : "notice n-warn";
    document.getElementById("gtStatusBanner").textContent =
      `已对齐事后 COCO GT (image_id=${phe.coco_image_id}, IoU>=0.50)：TP=${m.tp}, FP=${m.fp}, FN=${m.fn}, F1=${f1Disp} | ` +
      `身份验证=${phe.image_identity_verified ? "是" : "否"} (${phe.mapping_source}) — ${phe.identity_note || ""}`;
    document.getElementById("gtPerClassWrap").style.display = "block";
    const gcb = document.getElementById("liveGtPerClassBody");
    gcb.innerHTML = "";
    const pc = m.per_class || {};
    for (const cls of CLASSES) {
      const c = pc[cls] || {};
      const gtCnt = c.gt_count || 0;
      const cr = gtCnt === 0 ? "N/A (GT=0)" : (c.recall_display || pct(c.recall));
      const cf = gtCnt === 0 ? "N/A (GT=0)" : (c.f1_display || pct(c.f1));
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${cls}</td><td>${gtCnt}</td><td>${c.pred_count || 0}</td><td>${c.tp || 0}/${c.fp || 0}/${c.fn || 0}</td><td>${pct(c.precision)}</td><td>${cr}</td><td>${cf}</td>`;
      gcb.appendChild(tr);
    }
  } else {
    cards.push(["真值评价状态", "未评价（GT-free）"]);
    document.getElementById("gtPerClassWrap").style.display = "none";
  }
  document.getElementById("liveKpis").innerHTML = cards
    .map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");

  document.getElementById("liveNote").textContent =
    `${r.score_semantics_note} | 正常 EOS=${r.termination_v2.normal_eos}/${r.termination_v2.total_calls}，显式空数组=${r.termination_v2.explicit_empty}。`;

  const cb = document.getElementById("liveClassBody");
  cb.innerHTML = "";
  for (const cls of CLASSES) {
    const tr = document.createElement("tr");
    const preCnt = r.pre_verifier_class_counts ? r.pre_verifier_class_counts[cls] : "-";
    tr.innerHTML = `<td>${cls}</td><td>${r.class_counts[cls] || 0}</td><td>${preCnt}</td>`;
    cb.appendChild(tr);
  }

  const selCall = document.getElementById("selLiveCall");
  selCall.innerHTML = "";
  for (let i = 0; i < r.calls.length; i++) {
    const c = r.calls[i];
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${c.call_id} [${c.status}] obj=${c.parsed_count}`;
    selCall.appendChild(opt);
  }
  renderLiveSelectedCall();
  renderLiveItemsTable();
  renderLiveCanvas();
}

function renderLiveSelectedCall() {
  if (!LIVE_RESULT) return;
  const idx = Number(document.getElementById("selLiveCall").value || 0);
  const c = LIVE_RESULT.calls[idx];
  if (!c) return;
  document.getElementById("liveCallMeta").textContent =
    `call_id=${c.call_id} | 输入=${c.width}x${c.height} | tok=${c.generated_tokens} | GPU=${c.gpu_generate_ms}ms | VRAM=${c.cuda_max_reserved_mib}MiB`;
  document.getElementById("liveCallRaw").textContent = c.raw_output || "(空)";
}
document.getElementById("selLiveCall").onchange = renderLiveSelectedCall;

function getLiveRenderList() {
  if (!LIVE_RESULT) return [];
  const view = document.getElementById("liveOverlayView").value;
  if (view === "pre_verifier" && LIVE_RESULT.pre_verifier_candidates) {
    return LIVE_RESULT.pre_verifier_candidates.map(c => ({ ...c, _tag: c.kept ? "保留" : `剔除(${c.verifier_label || "异常"})` }));
  }
  const base = LIVE_RESULT.post_hoc_eval ? LIVE_RESULT.post_hoc_eval.predictions_matched : LIVE_RESULT.predictions;
  const list = base.map(p => ({ ...p, _tag: p.match_status || "保留" }));
  if (view === "with_rejected" && LIVE_RESULT.pre_verifier_candidates) {
    for (const c of LIVE_RESULT.pre_verifier_candidates) {
      if (!c.kept) list.push({ ...c, _rejected: true, _tag: `过滤:${c.verifier_label || "异常"}` });
    }
  }
  return list;
}

function renderLiveItemsTable() {
  const tbody = document.getElementById("liveItemsBody");
  tbody.innerHTML = "";
  if (!LIVE_RESULT) return;
  const rox = LIVE_RESULT.roi.offset_x, roy = LIVE_RESULT.roi.offset_y;
  for (const item of getLiveRenderList()) {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    const rx1 = Math.round(item.bbox_xyxy[0] - rox), ry1 = Math.round(item.bbox_xyxy[1] - roy);
    const rx2 = Math.round(item.bbox_xyxy[2] - rox), ry2 = Math.round(item.bbox_xyxy[3] - roy);
    const bcls = item._rejected || item._tag === "FP" || item._tag.startsWith("剔除") ? "b-rej" : "b-keep";
    tr.innerHTML = `<td><span class="badge ${bcls}">${item._tag}</span></td><td>${item.label}</td><td>[${rx1},${ry1},${rx2},${ry2}]</td>`;
    tr.onclick = () => {
      const pad = 260;
      LIVE_CROP = [
        Math.max(0, rx1 - pad),
        Math.max(0, ry1 - pad),
        Math.min(LIVE_RESULT.roi.width, rx2 + pad),
        Math.min(LIVE_RESULT.roi.height, ry2 + pad),
      ];
      renderLiveCanvas();
    };
    tbody.appendChild(tr);
  }
}

function renderLiveCanvas() {
  if (!LIVE_UPLOAD) return;
  const img = document.getElementById("liveImg");
  const svg = document.getElementById("liveSvg");
  const wrap = document.getElementById("liveCanvasWrap");
  const stage = document.getElementById("liveStage");
  const zoomVal = document.getElementById("liveZoom").value;

  const roiW = LIVE_UPLOAD.roi_width, roiH = LIVE_UPLOAD.roi_height;
  const rox = LIVE_UPLOAD.roi_offset_x, roy = LIVE_UPLOAD.roi_offset_y;
  const vx1 = LIVE_CROP ? LIVE_CROP[0] : 0;
  const vy1 = LIVE_CROP ? LIVE_CROP[1] : 0;
  const vx2 = LIVE_CROP ? LIVE_CROP[2] : roiW;
  const vy2 = LIVE_CROP ? LIVE_CROP[3] : roiH;
  const vw = Math.max(1, vx2 - vx1), vh = Math.max(1, vy2 - vy1);

  document.getElementById("liveCanvasTitle").textContent = LIVE_RESULT
    ? `检测结果：${LIVE_RESULT.safe_filename} [${LIVE_RESULT.model_key} / ${LIVE_RESULT.mode}]`
    : `预检 ROI 图像：${LIVE_UPLOAD.safe_filename} (${roiW}x${roiH}, ${LIVE_UPLOAD.tile_count} slices)`;
  document.getElementById("liveCropStatus").textContent = LIVE_CROP
    ? `局部裁剪视窗：[${vx1}, ${vy1}, ${vx2}, ${vy2}] (px)`
    : `完整 ROI 视野：${roiW} x ${roiH} (offset=[${rox}, ${roy}])`;

  const cropQ = LIVE_CROP ? `&crop=${vx1},${vy1},${vx2},${vy2}` : "";
  img.src = `/api/upload_preview/${encodeURIComponent(LIVE_UPLOAD.upload_id)}?max_width=${LIVE_CROP ? 1200 : 1800}${cropQ}`;
  const dispW = zoomVal === "fit" ? Math.max(460, wrap.clientWidth - 4) : Math.min(2800, Math.round(vw * Number(zoomVal)));
  const dispH = Math.round(dispW * (vh / vw));
  img.style.width = dispW + "px";
  img.style.height = dispH + "px";
  stage.style.width = dispW + "px";
  stage.style.height = dispH + "px";
  svg.setAttribute("viewBox", `${vx1} ${vy1} ${vw} ${vh}`);

  let elems = "";
  const sw = Math.max(2, vw / 420);
  const fs = Math.max(12, Math.round(vw / 75));
  const drawBox = (b, color, dash, text) => {
    const x1 = b[0] - rox, y1 = b[1] - roy, x2 = b[2] - rox, y2 = b[3] - roy;
    const w = Math.max(2, x2 - x1), h = Math.max(2, y2 - y1);
    elems += `<rect x="${x1}" y="${y1}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="${sw}" ${dash ? 'stroke-dasharray="' + (sw * 3) + ' ' + (sw * 2) + '"' : ""} />`;
    elems += `<text x="${x1}" y="${Math.max(vy1 + fs, y1 - 4)}" fill="${color}" font-size="${fs}" font-weight="bold" style="paint-order:stroke;stroke:#020617;stroke-width:${Math.max(3, fs / 4)}px;">${text}</text>`;
  };
  if (LIVE_RESULT) {
    if (LIVE_RESULT.post_hoc_eval) {
      for (const g of LIVE_RESULT.post_hoc_eval.gts) {
        if (g.match_status === "FN") drawBox(g.bbox_xyxy, "#facc15", true, `FN:${g.label}`);
      }
    }
    for (const item of getLiveRenderList()) {
      const col = item._rejected || item._tag.startsWith("剔除") ? "#fb7185"
        : item.match_status === "FP" ? "#ef4444" : "#38bdf8";
      drawBox(item.bbox_xyxy, col, Boolean(item._rejected || item.match_status === "FP"), `${item.label}`);
    }
  }
  svg.innerHTML = elems;
}

document.getElementById("liveOverlayView").onchange = () => { renderLiveItemsTable(); renderLiveCanvas(); };
document.getElementById("liveZoom").onchange = renderLiveCanvas;
document.getElementById("btnLiveResetCrop").onclick = () => { LIVE_CROP = null; renderLiveCanvas(); };

// Downloads
function dlExport(kind) {
  if (!LIVE_JOB_ID) return;
  const view = document.getElementById("liveOverlayView").value;
  window.location.href = `/api/jobs/${encodeURIComponent(LIVE_JOB_ID)}/export/${kind}?view=${encodeURIComponent(view)}`;
}
document.getElementById("btnExpJson").onclick = () => dlExport("json");
document.getElementById("btnExpPng").onclick = () => dlExport("overlay_png");
document.getElementById("btnExpMd").onclick = () => dlExport("summary_md");

// Optional Post-Hoc GT evaluation
document.getElementById("btnEvalGt").onclick = async () => {
  if (!LIVE_JOB_ID) return;
  const fIn = document.getElementById("fileGtCoco");
  const msgEl = document.getElementById("gtEvalMsg");
  msgEl.style.display = "none";
  if (!fIn.files || !fIn.files.length) {
    msgEl.textContent = "请先选择事后对比用的 COCO JSON 文件。";
    msgEl.style.display = "block";
    return;
  }
  try {
    const cocoObj = JSON.parse(await fIn.files[0].text());
    const imgIdRaw = document.getElementById("inpGtImageId").value.trim();
    const confirmManual = document.getElementById("chkGtManualConfirm").checked;
    const manualReason = document.getElementById("inpGtManualReason").value.trim();
    const body = { coco_gt: cocoObj, confirm_manual_mapping: confirmManual };
    if (imgIdRaw) body.target_image_id = Number(imgIdRaw);
    if (manualReason) body.manual_mapping_reason = manualReason;
    LIVE_RESULT = await apiJson(`/api/jobs/${encodeURIComponent(LIVE_JOB_ID)}/evaluate_gt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderLiveResult();
  } catch (err) {
    msgEl.textContent = err.message;
    msgEl.style.display = "block";
  }
};

// TAB 2: RESEARCH REPLAY
async function loadResearchOverview() {
  try {
    RES_OVERVIEW = await apiJson("/api/research/overview");
    if (!RES_OVERVIEW.available) {
      document.getElementById("researchUnavailable").textContent = RES_OVERVIEW.notice;
      document.getElementById("researchUnavailable").style.display = "block";
      return;
    }
    document.getElementById("researchContent").style.display = "block";
    const mc = RES_OVERVIEW.method_cards || {};
    let histHtml = "";
    for (const [k, ds] of Object.entries(RES_OVERVIEW.datasets)) {
      if (ds.split_group === "historical_or_dev") {
        const cc = ds.call_counts_per_model || {};
        histHtml += `<div>• <strong>${ds.label}</strong>：每模型 W=${cc.W || 2}，T=${cc.T || 0}（批次合计=${cc.batch_total || "-"}）` +
          (cc.B_other_crops ? `；Qwen3 二阶段裁剪=${cc.B_other_crops}` : "") + `</div>`;
      }
    }
    if (mc.dev_four_methods_s4_s20_163_gt) {
      const m = mc.dev_four_methods_s4_s20_163_gt.methods;
      histHtml += `<div style="margin-top:6px;">Dev (s4+s20, 163 GT) 预注册对比：T(F1=${pct(m.T.combined.f1)}) · A(F1=${pct(m.A.combined.f1)}) · <strong>B_other(TP=${m.B_other.combined.tp}, FP=${m.B_other.combined.fp}, F1=${pct(m.B_other.combined.f1)})</strong> · B_exact(F1=${pct(m.B_exact.combined.f1)})</div>`;
    }
    document.getElementById("histSummaryBox").innerHTML = histHtml;

    let hHtml = "";
    const hds = RES_OVERVIEW.datasets.heldout_blind_s19_s16;
    if (hds) {
      const cc = hds.call_counts_per_model || {};
      hHtml += `<div>• 单模型基线批次：W=${cc.W}，T=${cc.T}（合批=${cc.batch_total}；三模型共 ${cc.batch_total * 3} 次）</div>`;
      hHtml += `<div>• 冻结方案 <strong>Qwen3_T_plus_B_other</strong>：T=${cc.T} 次 + B_other 裁剪核验 ${cc.B_other_crops} 次（总计 ${cc.qwen3_T_plus_B_other_total} 次推理）</div>`;
    }
    if (mc.heldout_blind_s19_s16_107_gt) {
      const qb = mc.heldout_blind_s19_s16_107_gt.qwen3_T_B_other;
      hHtml += `<div style="margin-top:6px;color:#bae6fd;"><strong>最终盲测主指标 (IoU&gt;=0.50)：TP=${qb.tp} / FP=${qb.fp} / FN=${qb.fn} · Precision=${pct(qb.precision)} · Recall=${pct(qb.recall)} · F1=${pct(qb.f1)}</strong>（保留框 ${qb.predictions}/508，剔除 ${508 - qb.predictions} 个 other）</div>`;
    }
    document.getElementById("heldoutSummaryBox").innerHTML = hHtml;

    const selDs = document.getElementById("resDataset");
    selDs.innerHTML = "";
    for (const [k, ds] of Object.entries(RES_OVERVIEW.datasets)) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = `${ds.split_group === "heldout_blind" ? "[盲测保留] " : "[开发参考] "}${ds.label}`;
      selDs.appendChild(opt);
    }
    if (RES_OVERVIEW.datasets.heldout_blind_s19_s16) {
      selDs.value = "heldout_blind_s19_s16";
      document.getElementById("resModel").value = "qwen3";
      document.getElementById("resMode").value = "T_B_other";
    }
    populateResBoards();
    await loadResBoardView();
  } catch (err) {
    document.getElementById("researchUnavailable").textContent = "加载研究回放失败：" + err.message;
    document.getElementById("researchUnavailable").style.display = "block";
  }
}

function populateResBoards() {
  const dsKey = document.getElementById("resDataset").value;
  const ds = RES_OVERVIEW.datasets[dsKey];
  const selB = document.getElementById("resBoard");
  selB.innerHTML = "";
  for (const b of ds.boards) {
    const opt = document.createElement("option");
    opt.value = b.board_id;
    opt.textContent = `${b.board_id} (GT=${b.gt_count}, ${b.roi_width}x${b.roi_height})`;
    selB.appendChild(opt);
  }
}

async function loadResBoardView() {
  const errEl = document.getElementById("resErr");
  errEl.style.display = "none";
  RES_CROP = null;
  const dsKey = document.getElementById("resDataset").value;
  const bid = document.getElementById("resBoard").value;
  const mk = document.getElementById("resModel").value;
  const mode = document.getElementById("resMode").value;
  try {
    RES_CURRENT = await apiJson(`/api/research/board?dataset=${encodeURIComponent(dsKey)}&board=${encodeURIComponent(bid)}&model=${encodeURIComponent(mk)}&mode=${encodeURIComponent(mode)}`);
    renderResView();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.style.display = "block";
  }
}

function renderResView() {
  const r = RES_CURRENT;
  document.getElementById("resBoardTitle").textContent = `${r.dataset_label} · ${r.board_id} (image_id=${r.coco_image_id}) · ${r.model_key} [${r.mode}]`;
  const m = r.metrics;
  document.getElementById("resKpis").innerHTML = [
    ["TP / FP / FN", `${m.tp} / ${m.fp} / ${m.fn}`],
    ["Precision", pct(m.precision)],
    ["Recall", pct(m.recall)],
    ["F1", pct(m.f1)],
    ["GT 总数", m.total_gt],
    ["保留预测数", m.total_predictions],
  ].map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  const hasPre = Array.isArray(r.pre_verifier_candidates);
  document.getElementById("lblResPreT").style.display = hasPre ? "inline-flex" : "none";
  document.getElementById("resVerifierNote").textContent = r.verifier_summary
    ? `B_other 核验：输入 T 候选 ${r.verifier_summary.total_candidates} -> 保留 ${r.verifier_summary.kept_count}，剔除 ${r.verifier_summary.rejected_count}。`
    : "基线输出（无二阶段 B_other 过滤）。";

  const selC = document.getElementById("selResCall");
  selC.innerHTML = "";
  for (let i = 0; i < r.calls.length; i++) {
    const c = r.calls[i];
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${c.call_id} [${c.status}] obj=${c.parsed_count}`;
    selC.appendChild(opt);
  }
  renderResCall();

  const tbody = document.getElementById("resItemsBody");
  tbody.innerHTML = "";
  const rox = r.roi.offset_x, roy = r.roi.offset_y;
  const rows = [];
  for (const p of r.predictions) rows.push({ k: p.match_status, l: p.label, b: p.bbox_xyxy, iou: p.matched_iou });
  for (const g of r.gts) if (g.match_status === "FN") rows.push({ k: "FN", l: g.label, b: g.bbox_xyxy, iou: null });
  for (const it of rows) {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    const rx1 = Math.round(it.b[0] - rox), ry1 = Math.round(it.b[1] - roy);
    const rx2 = Math.round(it.b[2] - rox), ry2 = Math.round(it.b[3] - roy);
    const bcls = it.k === "TP" ? "b-tp" : (it.k === "FP" ? "b-fp" : "b-fn");
    tr.innerHTML = `<td><span class="badge ${bcls}">${it.k}</span></td><td>${it.l}</td><td>${it.iou ? "IoU=" + it.iou + " " : ""}[${rx1},${ry1},${rx2},${ry2}]</td>`;
    tr.onclick = () => {
      RES_CROP = [Math.max(0, rx1 - 260), Math.max(0, ry1 - 260), Math.min(r.roi.width, rx2 + 260), Math.min(r.roi.height, ry2 + 260)];
      renderResCanvas();
    };
    tbody.appendChild(tr);
  }
  document.getElementById("resProvPre").textContent = JSON.stringify({
    replay_sha256: r.provenance.replay_report?.sha256,
    gt_coco_sha256: r.provenance.gt_coco?.sha256,
    raw_jsonl_sha256: r.raw_provenance[r.model_key]?.sha256,
    roi_sha256: r.roi.sha256,
  }, null, 2);
  renderResCanvas();
}

function renderResCall() {
  if (!RES_CURRENT) return;
  const c = RES_CURRENT.calls[Number(document.getElementById("selResCall").value || 0)];
  if (!c) return;
  document.getElementById("resCallMeta").textContent = `call_id=${c.call_id} | tokens=${c.generated_tokens} | GPU=${c.gpu_generate_ms}ms | VRAM=${c.cuda_max_reserved_mib}MiB`;
  document.getElementById("resCallRaw").textContent = c.raw_output || "(空)";
}
document.getElementById("selResCall").onchange = renderResCall;

function renderResCanvas() {
  if (!RES_CURRENT) return;
  const r = RES_CURRENT;
  const img = document.getElementById("resImg");
  const svg = document.getElementById("resSvg");
  const wrap = document.getElementById("resCanvasWrap");
  const stage = document.getElementById("resStage");
  const zoomVal = document.getElementById("resZoom").value;
  const rox = r.roi.offset_x, roy = r.roi.offset_y;
  const vx1 = RES_CROP ? RES_CROP[0] : 0, vy1 = RES_CROP ? RES_CROP[1] : 0;
  const vx2 = RES_CROP ? RES_CROP[2] : r.roi.width, vy2 = RES_CROP ? RES_CROP[3] : r.roi.height;
  const vw = Math.max(1, vx2 - vx1), vh = Math.max(1, vy2 - vy1);
  const cropQ = RES_CROP ? `&crop=${vx1},${vy1},${vx2},${vy2}` : "";
  img.src = `/api/research/image?dataset=${encodeURIComponent(r.dataset_key)}&board=${encodeURIComponent(r.board_id)}&max_width=${RES_CROP ? 1200 : 1800}${cropQ}`;
  const dispW = zoomVal === "fit" ? Math.max(460, wrap.clientWidth - 4) : Math.min(2800, Math.round(vw * Number(zoomVal)));
  const dispH = Math.round(dispW * (vh / vw));
  img.style.width = dispW + "px"; img.style.height = dispH + "px";
  stage.style.width = dispW + "px"; stage.style.height = dispH + "px";
  svg.setAttribute("viewBox", `${vx1} ${vy1} ${vw} ${vh}`);

  const sTP = document.getElementById("chkResTP").checked;
  const sFP = document.getElementById("chkResFP").checked;
  const sFN = document.getElementById("chkResFN").checked;
  const sPre = document.getElementById("chkResPreT").checked;
  const sw = Math.max(2, vw / 420), fs = Math.max(12, Math.round(vw / 75));
  let elems = "";
  const drawBox = (b, col, dash, txt) => {
    const x1 = b[0] - rox, y1 = b[1] - roy, w = Math.max(2, b[2] - b[0]), h = Math.max(2, b[3] - b[1]);
    elems += `<rect x="${x1}" y="${y1}" width="${w}" height="${h}" fill="none" stroke="${col}" stroke-width="${sw}" ${dash ? 'stroke-dasharray="' + (sw*3) + ' ' + (sw*2) + '"' : ""} />`;
    elems += `<text x="${x1}" y="${Math.max(vy1 + fs, y1 - 4)}" fill="${col}" font-size="${fs}" font-weight="bold" style="paint-order:stroke;stroke:#020617;stroke-width:${Math.max(3,fs/4)}px;">${txt}</text>`;
  };
  if (sPre && r.pre_verifier_candidates) {
    for (const c of r.pre_verifier_candidates) if (!c.kept) drawBox(c.bbox_xyxy, "#fb7185", true, `Filtered:${c.label}`);
  }
  for (const g of r.gts) if (g.match_status === "FN" && sFN) drawBox(g.bbox_xyxy, "#facc15", true, `FN:${g.label}`);
  for (const p of r.predictions) {
    if (p.match_status === "TP" && sTP) drawBox(p.bbox_xyxy, "#38bdf8", false, `TP:${p.label}`);
    if (p.match_status === "FP" && sFP) drawBox(p.bbox_xyxy, "#ef4444", true, `FP:${p.label}`);
  }
  svg.innerHTML = elems;
}

document.getElementById("resDataset").onchange = () => { populateResBoards(); loadResBoardView(); };
document.getElementById("resBoard").onchange = loadResBoardView;
document.getElementById("resModel").onchange = loadResBoardView;
document.getElementById("resMode").onchange = loadResBoardView;
document.getElementById("resZoom").onchange = renderResCanvas;
document.getElementById("btnResResetCrop").onclick = () => { RES_CROP = null; renderResCanvas(); };
for (const id of ["chkResTP", "chkResFP", "chkResFN", "chkResPreT"]) document.getElementById(id).onchange = renderResCanvas;

async function refreshRecentJobs() {
  try {
    const d = await apiJson("/api/jobs");
    const sel = document.getElementById("selRecentJob");
    const cur = sel.value;
    sel.innerHTML = '<option value="">（可从本地工作区重新载入近期任务）</option>';
    for (const j of d.jobs || []) {
      const opt = document.createElement("option");
      opt.value = j.job_id;
      opt.textContent = `${j.job_id} | ${j.model_key}/${j.mode} [${j.status}]`;
      sel.appendChild(opt);
    }
    if (cur) sel.value = cur;
  } catch (_) {}
}

document.getElementById("btnLoadRecentJob").onclick = async () => {
  const jid = document.getElementById("selRecentJob").value;
  if (!jid) return;
  const errBox = document.getElementById("jobErrorBox");
  errBox.style.display = "none";
  try {
    const st = await apiJson(`/api/jobs/${encodeURIComponent(jid)}`);
    LIVE_JOB_ID = st.job_id;
    document.getElementById("jobProgressCard").style.display = "block";
    updateProgressUi(st);
    if (st.has_result) {
      LIVE_RESULT = await apiJson(`/api/jobs/${encodeURIComponent(jid)}/result`);
      LIVE_CROP = null;
      LIVE_UPLOAD = {
        upload_id: LIVE_RESULT.upload_id,
        safe_filename: LIVE_RESULT.safe_filename,
        roi_width: LIVE_RESULT.roi.width,
        roi_height: LIVE_RESULT.roi.height,
        roi_offset_x: LIVE_RESULT.roi.offset_x,
        roi_offset_y: LIVE_RESULT.roi.offset_y,
        tile_count: LIVE_RESULT.roi.tile_count,
      };
      document.getElementById("btnEvalGt").disabled = false;
      renderLiveResult();
    } else if (["queued", "running_base_inference", "preparing_verifier", "running_verifier"].includes(st.status)) {
      if (LIVE_POLL_TIMER) clearInterval(LIVE_POLL_TIMER);
      LIVE_POLL_TIMER = setInterval(pollJob, 800);
    } else {
      errBox.textContent = st.error_message || ("任务状态：" + st.status);
      errBox.style.display = "block";
    }
  } catch (err) {
    errBox.textContent = "载入任务失败：" + err.message;
    errBox.style.display = "block";
  }
};

syncModeOptions();
refreshRecentJobs();
</script>
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
            elif path == "/favicon.ico":
                self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            elif path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "research_available": bool(self.service.research_store and self.service.research_store.available)})
            elif path.startswith("/api/upload_preview/"):
                uid = urllib.parse.unquote(path[len("/api/upload_preview/"):])
                max_w = int((qs.get("max_width") or ["1600"])[0])
                crop = (qs.get("crop") or [None])[0]
                jpg = self.service.render_upload_roi_jpeg(upload_id=uid, max_width=max_w, crop=crop)
                self._send_bytes(HTTPStatus.OK, jpg, "image/jpeg")
            elif path == "/api/jobs":
                self._send_json(HTTPStatus.OK, {"jobs": self.service.list_jobs()})
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
            elif path == "/api/jobs":
                body = json.loads(self._read_body().decode("utf-8"))
                out = self.service.submit_job(
                    upload_id=str(body.get("upload_id") or ""),
                    model_key=str(body.get("model_key") or ""),
                    mode=str(body.get("mode") or ""),
                )
                self._send_json(HTTPStatus.ACCEPTED, out)
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
        r_store = ResearchReplayStore(results_index_path) if results_index_path else ResearchReplayStore(None)
        service = LiveDemoService(
            workspace_root=workspace_root,
            runtime_config_path=runtime_config_path,
            research_store=r_store,
        )
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
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = {}
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

    ws_raw = args.workspace_root if args.workspace_root is not None else cfg.get("workspace_root")
    if not ws_raw:
        raise SystemExit("Missing workspace_root: provide --workspace-root or set 'workspace_root' in --config JSON.")

    rt_raw = args.runtime_config if args.runtime_config is not None else cfg.get("runtime_config")
    idx_raw = args.results_index if args.results_index is not None else cfg.get("results_index")
    host = args.host if args.host is not None else str(cfg.get("host") or "127.0.0.1")
    port = args.port if args.port is not None else int(cfg.get("port") if cfg.get("port") is not None else 7862)
    if args.no_port_fallback:
        allow_fallback = False
    elif "allow_port_fallback" in cfg:
        allow_fallback = bool(cfg["allow_port_fallback"])
    else:
        allow_fallback = True

    return {
        "workspace_root": pathlib.Path(str(ws_raw)),
        "runtime_config_path": pathlib.Path(str(rt_raw)) if rt_raw else None,
        "results_index_path": pathlib.Path(str(idx_raw)) if idx_raw else None,
        "host": host,
        "port": port,
        "allow_port_fallback": allow_fallback,
    }


def main(argv: Optional[List[str]] = None) -> int:
    opts = resolve_server_launch_options(argv)
    srv = create_demo_server(
        workspace_root=opts["workspace_root"],
        runtime_config_path=opts["runtime_config_path"],
        results_index_path=opts["results_index_path"],
        host=opts["host"],
        port=opts["port"],
        allow_port_fallback=opts["allow_port_fallback"],
    )
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
