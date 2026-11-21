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
  .wrap { padding: 14px 20px; max-width: 1760px; margin: 0 auto; }
  .grid-main { display: grid; grid-template-columns: minmax(350px, 430px) minmax(0, 1fr); gap: 14px; align-items: start; }
  @media (max-width: 1100px) { .grid-main { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin-bottom: 12px; }
  .card h2 { margin: 0 0 10px 0; font-size: 14px; color: #e2e8f0; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  label.fld { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); width: 100%; margin-bottom: 8px; }
  select, input[type="file"], input[type="number"], input[type="text"], button { background: var(--panel-2); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; font-size: 12.5px; }
  button { cursor: pointer; font-weight: 600; }
  button.primary { background: #0284c7; border-color: #38bdf8; color: #fff; }
  button.primary:hover:not(:disabled) { background: #0369a1; }
  button.accent-purple { background: #581c87; border-color: #c084fc; color: #f3e8ff; }
  button.danger { background: #7f1d1d; border-color: #ef4444; color: #fee2e2; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .notice { padding: 9px 12px; border-radius: 8px; font-size: 12px; line-height: 1.5; margin-bottom: 10px; }
  .n-info { background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.35); color: #bae6fd; }
  .n-warn { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); color: #fde68a; }
  .n-err { background: rgba(239,68,68,0.14); border: 1px solid rgba(239,68,68,0.5); color: #fecaca; }
  .n-ok { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.4); color: #bbf7d0; }
  .n-purple { background: rgba(192,132,252,0.12); border: 1px solid rgba(192,132,252,0.45); color: #e9d5ff; }
  .progress-track { width: 100%; height: 10px; background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px; overflow: hidden; margin: 8px 0; }
  .progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #0ea5e9, #22c55e); transition: width 0.25s ease; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-bottom: 10px; }
  .kpi { background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .kpi .k { font-size: 11px; color: var(--muted); }
  .kpi .v { font-size: 15px; font-weight: 700; margin-top: 2px; }
  .canvas-wrap { position: relative; width: 100%; overflow: auto; max-height: 64vh; background: #020617; border: 1px solid var(--line); border-radius: 8px; }
  .stage { position: relative; display: inline-block; transform-origin: top left; }
  .stage img { display: block; max-width: none; }
  .stage svg { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }
  .tbl-wrap { max-height: 230px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 6px 8px; border-bottom: 1px solid #1e293b; text-align: left; }
  th { background: #0f172a; color: #cbd5e1; position: sticky; top: 0; z-index: 1; }
  tr.clickable:hover { background: #1e293b; cursor: pointer; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .b-keep { background: rgba(56,189,248,0.2); color: #7dd3fc; }
  .b-ok { background: rgba(34,197,94,0.2); color: #86efac; }
  .b-zero { background: rgba(245,158,11,0.22); color: #fde68a; }
  .b-rej { background: rgba(244,63,94,0.2); color: #fda4af; }
  .b-wait { background: rgba(148,163,184,0.2); color: #cbd5e1; }
  .b-run { background: rgba(14,165,233,0.25); color: #7dd3fc; }
  .b-tp { background: rgba(56,189,248,0.2); color: #7dd3fc; }
  .b-fp { background: rgba(239,68,68,0.2); color: #fca5a5; }
  .b-fn { background: rgba(245,158,11,0.2); color: #fde047; }
  pre.raw-pre { white-space: pre-wrap; word-break: break-word; max-height: 170px; overflow: auto; background: #020617; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-size: 11.5px; margin: 6px 0 0 0; }
  .research-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; }
  @media (max-width: 1050px) { .research-grid { grid-template-columns: 1fr; } }
  .model-check-item { display: flex; flex-direction: column; gap: 3px; padding: 8px 10px; background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 6px; }
  .model-check-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 600; }
  .model-check-sub { font-size: 11px; color: var(--muted); padding-left: 20px; }
  .cmp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 12px; margin-top: 10px; }
  .cmp-col { background: var(--panel-2); border: 1px solid var(--line); border-radius: 9px; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  .cmp-col.active-focus { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(56,189,248,0.45); }
  .cmp-canvas-box { position: relative; width: 100%; overflow: auto; max-height: 300px; background: #020617; border: 1px solid var(--line); border-radius: 6px; }
  .cls-pill { display: inline-flex; align-items: center; gap: 5px; padding: 4px 8px; border-radius: 6px; background: var(--panel-2); border: 1px solid var(--line); font-size: 12px; cursor: pointer; user-select: none; }
  .cls-swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  .cls-mini-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 4px; font-size: 11.5px; }
  .cls-mini-cell { background: #131e33; border: 1px solid #1e293b; border-radius: 4px; padding: 3px 6px; display: flex; justify-content: space-between; }
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>基于开源视觉语言大模型的电子元器件检测系统</h1>
      <div class="sub">FICS-PCB 6 类目标（电容/电阻/电感/晶体管/二极管/集成电路） · 上传新图无 GT 多模型同图串行对比与历史研究回放严格隔离</div>
    </div>
    <div class="tabs">
      <button id="tabBtnLive" class="tab-btn active" type="button">上传并检测（新图单模型 / 三模型同图对比）</button>
      <button id="tabBtnResearch" class="tab-btn" type="button">历史研究回放（8 板只读归档）</button>
    </div>
  </div>

  <!-- TAB 1: LIVE GT-FREE WORKFLOW & MULTI-MODEL COMPARISON -->
  <div id="tabLive" class="wrap">
    <div class="grid-main">
      <div>
        <div class="card">
          <h2>1. 上传真实 PCB 图像（默认入口 · 无需 Ground Truth）</h2>
          <div class="notice n-info">
            直接上传待测单帧 PCB 图像（PNG / JPG / TIFF），不依赖选择预置板号或提供人工标注。系统自动执行分辨率/空白页预检、白边 ROI 裁切（阈值 &lt;248 + 32px）与 1024px 切片预算校核，<strong>绝不降低输入分辨率或删减切片</strong>。
          </div>
          <label class="fld">选择 PCB 图像文件（PNG / JPG / TIFF）
            <input id="fileBoard" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp" />
          </label>
          <div class="row">
            <button id="btnUpload" class="primary" type="button">上传并预检图像尺寸 / ROI / 切片</button>
          </div>
          <div id="uploadMsg" class="notice n-warn" style="display:none;margin-top:8px;"></div>
          <div id="uploadMetaBox" style="margin-top:10px;font-size:12px;color:var(--muted);line-height:1.55;"></div>
        </div>

        <div class="card">
          <h2>2. 参测模型与同图串行对比设置（单 GPU 互斥队列）</h2>
          <label class="fld">同配置基线检测模式（对比默认采用三模型通用的整图 ROI 模式 W）
            <select id="liveMode">
              <option value="W" selected>W（整图 ROI 推理 · 三模型通用默认对比基线）</option>
              <option value="T">T（1024px 切片推理 + 0.50 NMS · 三模型通用）</option>
              <option value="T_B_other">T + B_other（切片+背景核验 · 冻结方法 · 仅限 Qwen3 单模型增强）</option>
            </select>
          </label>
          <div id="modeHint" style="font-size:11.5px;color:var(--muted);margin-bottom:8px;">
            当前为整图 ROI 模式（W）：每模型 1 次整板推理，三个模型各自采用其冻结的原生视觉预处理器（动态分辨率 vs 384×384 固定正方形缩放）。
          </div>

          <div style="font-size:12px;color:#cbd5e1;font-weight:600;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;">
            <span>勾选同图串行对比模型（按队列逐个独占 GPU 运行）：</span>
            <span class="row" style="gap:4px;">
              <button id="btnSelAllModels" type="button" style="padding:3px 7px;font-size:11px;">全选三模型</button>
              <button id="btnSelOnlyFocus" type="button" style="padding:3px 7px;font-size:11px;">仅选焦点模型</button>
            </span>
          </div>

          <div class="model-check-item">
            <div class="model-check-top">
              <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                <input type="checkbox" id="chkModel_qwen2" value="qwen2" checked />
                <span>Qwen2-VL-2B-Instruct</span>
              </label>
              <span id="stBadge_qwen2" class="badge b-wait">未运行</span>
            </div>
            <div class="model-check-sub">原生预处理：动态分辨率预算 (min_pixels=65,536, max_pixels=200,704, SDPA)</div>
          </div>

          <div class="model-check-item">
            <div class="model-check-top">
              <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                <input type="checkbox" id="chkModel_deepseek" value="deepseek" checked />
                <span>deepseek-vl-1.3b-chat（v1）</span>
              </label>
              <span id="stBadge_deepseek" class="badge b-wait">未运行</span>
            </div>
            <div class="model-check-sub">原生预处理：固定 384×384 正方形缩放 + [127,127,127] 灰边填充</div>
          </div>

          <div class="model-check-item">
            <div class="model-check-top">
              <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                <input type="checkbox" id="chkModel_qwen3" value="qwen3" checked />
                <span>Qwen3-VL-2B-Instruct</span>
              </label>
              <span id="stBadge_qwen3" class="badge b-wait">未运行</span>
            </div>
            <div class="model-check-sub">原生预处理：动态分辨率预算 (shortest_edge=65,536, longest_edge=200,704, SDPA + 独立依赖覆盖)</div>
          </div>

          <label id="lblAddonBOther" style="display:flex;gap:6px;align-items:flex-start;font-size:11.5px;color:#e9d5ff;background:rgba(192,132,252,0.1);border:1px solid rgba(192,132,252,0.35);padding:7px 9px;border-radius:7px;margin:8px 0;">
            <input type="checkbox" id="chkAddonQwen3BOther" />
            <span>可选附加：队列末尾追加运行 <strong>Qwen3-VL-2B-Instruct [T + B_other 冻结二次核验]</strong> 作为单独增强对照（独立分栏展示，不混入 W/T 同配置基线列；建议切片数 &le;24 时启用）</span>
          </label>

          <label class="fld" style="margin-top:8px;">单模型运行 / 下方主画布当前焦点模型
            <select id="liveModel">
              <option value="qwen3">Qwen3-VL-2B-Instruct</option>
              <option value="qwen2">Qwen2-VL-2B-Instruct</option>
              <option value="deepseek">deepseek-vl-1.3b-chat（v1）</option>
            </select>
          </label>

          <div class="row" style="margin-top:6px;">
            <button id="btnStartCompare" class="primary" type="button" disabled>启动所选模型同图串行实跑</button>
            <button id="btnStartJob" type="button" disabled>仅运行当前焦点单模型</button>
            <button id="btnCancelJob" class="danger" type="button" disabled>安全取消当前运行 / 队列</button>
          </div>

          <div class="row" style="margin-top:8px;">
            <select id="selRecentCmp" style="flex:1;min-width:170px;"><option value="">（载入近期同图多模型对比记录）</option></select>
            <button id="btnLoadRecentCmp" type="button">载入对比</button>
          </div>
          <div class="row" style="margin-top:6px;">
            <select id="selRecentJob" style="flex:1;min-width:170px;"><option value="">（载入近期单模型任务记录）</option></select>
            <button id="btnLoadRecentJob" type="button">载入单任务</button>
          </div>

          <div id="jobErrorBox" class="notice n-err" style="display:none;margin-top:8px;"></div>
          <div id="pollRecoveryBar" class="notice n-warn" style="display:none;margin-top:8px;">
            <div id="pollRecoveryMsg" style="margin-bottom:6px;"></div>
            <button id="btnResumePoll" type="button" style="padding:4px 9px;font-size:11.5px;">立即重试状态刷新</button>
          </div>
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
          <h2>3. 真值评价状态与事后可选 COCO GT 校验</h2>
          <div id="gtStatusBanner" class="notice n-warn">
            <strong>模型预测 / 未做精度评价（GT-free）</strong>：当前未提供人工标注真值，仅展示模型预测位置与类别计数；预测框数不代表板上真实器件数，不计算准确率/F1，<code>score=1.0</code> 仅为离散坐标占位符而非 100% 置信度。
          </div>
          <details>
            <summary style="font-size:12px;color:var(--muted);cursor:pointer;">展开可选功能：事后上传该板对应的 COCO GT JSON 进行对齐评价</summary>
            <div style="margin-top:8px;">
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
                <input id="inpGtManualReason" type="text" placeholder="例如：原图重命名为测试编号" />
              </label>
              <button id="btnEvalGt" type="button" disabled>对当前焦点模型提交事后 GT 评价</button>
              <div id="gtEvalMsg" class="notice n-err" style="display:none;margin-top:8px;"></div>
            </div>
          </details>
        </div>
      </div>

      <div>
        <!-- GLOBAL VISUAL FILTER, LEGEND & COORDINATE SPACE BAR -->
        <div class="card">
          <div class="row" style="justify-content:space-between;margin-bottom:8px;">
            <strong style="font-size:13.5px;color:#e0f2fe;">同图交互控制栏：坐标视野 / 缩放 / 6 类元器件颜色图例与筛选</strong>
            <div class="row">
              <label style="font-size:12px;color:var(--muted);">图像与坐标基准：
                <select id="liveCoordSpace">
                  <option value="roi" selected>ROI 裁切视图（相对 ROI 坐标）</option>
                  <option value="orig">完整原图视图（原图全局坐标 + ROI 边界）</option>
                </select>
              </label>
              <label style="font-size:12px;color:var(--muted);">主画布层：
                <select id="liveOverlayView">
                  <option value="kept">核验后保留 / 最终预测框</option>
                  <option value="pre_verifier">原 T 候选（核验前）</option>
                  <option value="with_rejected">叠加显示被 B_other 过滤候选</option>
                </select>
              </label>
              <label style="font-size:12px;color:var(--muted);">缩放：
                <select id="liveZoom">
                  <option value="fit">自适应视野</option>
                  <option value="1">1.0x</option>
                  <option value="2">2.0x</option>
                </select>
              </label>
              <button id="btnLiveResetCrop" type="button">重置全图视野</button>
            </div>
          </div>

          <div class="row" id="classFilterBar" style="gap:6px;">
            <span style="font-size:12px;color:var(--muted);margin-right:2px;">6 类筛选与图例：</span>
            <label class="cls-pill"><input type="checkbox" class="chk-cls" value="capacitor" checked /><span class="cls-swatch" style="background:#38bdf8;"></span>电容 (capacitor)</label>
            <label class="cls-pill"><input type="checkbox" class="chk-cls" value="resistor" checked /><span class="cls-swatch" style="background:#22c55e;"></span>电阻 (resistor)</label>
            <label class="cls-pill"><input type="checkbox" class="chk-cls" value="inductor" checked /><span class="cls-swatch" style="background:#facc15;"></span>电感 (inductor)</label>
            <label class="cls-pill"><input type="checkbox" class="chk-cls" value="transistor" checked /><span class="cls-swatch" style="background:#c084fc;"></span>晶体管 (transistor)</label>
            <label class="cls-pill"><input type="checkbox" class="chk-cls" value="diode" checked /><span class="cls-swatch" style="background:#fb923c;"></span>二极管 (diode)</label>
            <label class="cls-pill"><input type="checkbox" class="chk-cls" value="ic" checked /><span class="cls-swatch" style="background:#2dd4bf;"></span>集成电路 (ic)</label>
            <button id="btnClsAll" type="button" style="padding:3px 7px;font-size:11px;">全选类别</button>
            <button id="btnClsNone" type="button" style="padding:3px 7px;font-size:11px;">清空筛选</button>
          </div>
          <div id="liveCropStatus" style="font-size:12px;color:var(--muted);margin-top:6px;">等待上传图像…</div>
        </div>

        <!-- SAME-IMAGE MULTI-MODEL SIDE-BY-SIDE COMPARISON SECTION -->
        <div class="card" id="compareSectionCard" style="display:none;">
          <div class="row" style="justify-content:space-between;margin-bottom:8px;">
            <div>
              <h2 style="margin:0;border:none;padding:0;">同图多模型并排对比面板（严格绑定同一上传 ID 与原图 SHA-256）</h2>
              <div id="cmpIdentityBanner" style="font-size:11.5px;color:#93c5fd;margin-top:3px;">-</div>
            </div>
            <div class="row">
              <button id="btnExpCmpMd" class="primary" type="button">导出同图对比报告 (.md)</button>
              <button id="btnExpCmpJson" type="button">导出同图对比 JSON (.json)</button>
            </div>
          </div>

          <div class="notice n-info" style="margin-bottom:8px;">
            <strong>状态与零框说明</strong>：界面严格区分“<strong>完成（检出预测框）</strong>”、“<strong>无有效框</strong>（模型输出空数组 <code>[]</code> 或达 token 上限未闭合，<strong>不等于电路板上没有元器件</strong>）”与“<strong>运行失败 / 已取消</strong>（不生成伪造空成绩）”。无真值时仅展示模型预测，不输出准确率。
          </div>

          <div class="tbl-wrap" style="max-height:195px;margin-bottom:10px;">
            <table>
              <thead>
                <tr>
                  <th>模型实际名称</th>
                  <th>模式</th>
                  <th>运行/检测状态</th>
                  <th>耗时</th>
                  <th>预测总框数 (当前筛选可见)</th>
                  <th>电容</th>
                  <th>电阻</th>
                  <th>电感</th>
                  <th>晶体管</th>
                  <th>二极管</th>
                  <th>集成电路</th>
                  <th>终止与解析详情</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody id="compareSummaryBody"></tbody>
            </table>
          </div>

          <div id="compareCardsGrid" class="cmp-grid"></div>

          <div id="compareAddonWrap" style="display:none;margin-top:12px;padding-top:10px;border-top:1px dashed #c084fc;">
            <div class="notice n-purple" style="margin-bottom:8px;">
              <strong>单独冻结增强结果（非同配置基线列）</strong>：以下为 <code>Qwen3-VL-2B-Instruct [T + B_other]</code>（切片 + 10% 外扩裁剪背景核验）在同一张图上的独立增强输出，按模式分栏展示，不混入上方同配置基线对比列。
            </div>
            <div id="compareAddonGrid" class="cmp-grid"></div>
          </div>
        </div>

        <!-- MAIN LARGE CANVAS & DETAILED OBJECT TABLE FOR FOCUSED MODEL -->
        <div class="card">
          <div class="row" style="justify-content:space-between;margin-bottom:8px;">
            <strong id="liveCanvasTitle" style="font-size:14px;">主检视画布（请先上传图像并启动推理）</strong>
            <div class="row" id="focusModelSwitchBar" style="gap:6px;"></div>
          </div>
          <div id="liveCanvasWrap" class="canvas-wrap">
            <div id="liveStage" class="stage">
              <img id="liveImg" alt="Live PCB Preview" />
              <svg id="liveSvg" preserveAspectRatio="none"></svg>
            </div>
          </div>
        </div>

        <div class="card" id="liveResultCard" style="display:none;">
          <div class="row" style="justify-content:space-between;margin-bottom:10px;">
            <h2 id="liveResultHeading" style="margin:0;border:none;padding:0;">当前焦点模型检测结果详情与单模型导出</h2>
            <div class="row">
              <button id="btnExpJson" type="button">下载当前模型 JSON</button>
              <button id="btnExpPng" type="button">下载当前模型叠框 PNG</button>
              <button id="btnExpMd" type="button">下载当前模型摘要 (.md)</button>
            </div>
          </div>
          <div id="liveKpis" class="kpis"></div>
          <div id="liveNote" style="font-size:12px;color:var(--muted);margin-bottom:8px;"></div>
          <div class="row" style="gap:12px;align-items:flex-start;">
            <div style="flex:1;min-width:250px;">
              <div style="font-size:12px;font-weight:700;margin-bottom:4px;">6 类检出计数（中文 / 英文类名）</div>
              <div class="tbl-wrap" style="max-height:200px;">
                <table><thead><tr><th>元器件类别</th><th>最终保留数</th><th>原 T 候选数</th></tr></thead><tbody id="liveClassBody"></tbody></table>
              </div>
            </div>
            <div style="flex:1.6;min-width:340px;">
              <div style="font-size:12px;font-weight:700;margin-bottom:4px;">预测目标列表（点击任意行可同步放大定位主画布与并排对比画布）</div>
              <div class="tbl-wrap" style="max-height:200px;">
                <table><thead><tr><th>状态</th><th>类别</th><th>ROI 坐标 [x1,y1,x2,y2]</th><th>原图全局坐标 [x1,y1,x2,y2]</th></tr></thead><tbody id="liveItemsBody"></tbody></table>
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
                <select id="selLiveCall" style="max-width:360px;"></select>
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
    <div class="notice n-purple" style="font-size:12.5px;">
      <strong>【历史研究回放 · 只读归档专区】</strong>本页展示已冻结的 8 块 FICS-PCB 正式实验板卡（6 板开发/历史参考 268 GT + 2 板盲测保留集 <code>s19+s16</code> 107 GT）归档记录，<strong>与“上传并检测”标签页中的新图实时 GPU 推理严格隔离</strong>。
    </div>
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
              <option value="deepseek">deepseek-vl-1.3b-chat（v1）</option>
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
const CLASS_CN = {
  capacitor: "电容",
  resistor: "电阻",
  inductor: "电感",
  transistor: "晶体管",
  diode: "二极管",
  ic: "集成电路",
};
const CLASS_COLORS = {
  capacitor: "#38bdf8",
  resistor: "#22c55e",
  inductor: "#facc15",
  transistor: "#c084fc",
  diode: "#fb923c",
  ic: "#2dd4bf",
};
const MODEL_NAMES = {
  qwen2: "Qwen2-VL-2B-Instruct",
  deepseek: "deepseek-vl-1.3b-chat（v1）",
  qwen3: "Qwen3-VL-2B-Instruct",
};

let LIVE_UPLOAD = null;
let LIVE_JOB_ID = null;
let LIVE_RESULT = null;
let LIVE_COMPARISON_ID = null;
let LIVE_COMPARISON = null;
let LIVE_CROP = null; // always stored in ROI coordinates [rx1, ry1, rx2, ry2]
let LIVE_POLL_TIMER = null;

// Monotonic viewing-session / request-generation tokens to guarantee stale async responses
// can never overwrite UI state, flip buttons, mix exports, or stop another session's polling timer.
let LIVE_SESSION_SEQ = 0;
let LIVE_FOCUS_SEQ = 0;
let LIVE_UPLOAD_SEQ = 0;
let LIVE_ACTIVE_SESSION = {
  sessionId: 0,
  kind: "idle", // "idle" | "upload" | "comparison" | "job"
  uploadId: null,
  uploadSha256: null,
  comparisonId: null,
  jobId: null,
  timerId: null,
  abortController: null,
  pollFailures: 0,
  isRunning: false,
};

let RES_OVERVIEW = null;
let RES_CURRENT = null;
let RES_CROP = null;
let RES_REQUEST_SEQ = 0;

function _makeAbortController() {
  return (typeof AbortController !== "undefined") ? new AbortController() : null;
}

function stopSessionTimer(session, explicitTimerId = null) {
  const tid = (explicitTimerId !== null && explicitTimerId !== undefined)
    ? explicitTimerId
    : (session ? session.timerId : null);
  if (tid !== null && tid !== undefined) {
    clearInterval(tid);
    if (session && session.timerId === tid) {
      session.timerId = null;
    }
    if (LIVE_POLL_TIMER === tid) {
      LIVE_POLL_TIMER = null;
    }
  }
}

const TERMINAL_JOB_STATUSES = ["completed", "failed", "cancelled", "incomplete"];
function isJobTerminalStatus(st) {
  return Boolean(st && TERMINAL_JOB_STATUSES.includes(st));
}
function isComparisonTerminalStatus(st) {
  return Boolean(st && st !== "running");
}

function hidePollRecovery() {
  const bar = document.getElementById("pollRecoveryBar");
  if (bar) bar.style.display = "none";
}

function showPollRecovery(session, message, paused) {
  if (!isLiveSessionCurrent(session)) return;
  const bar = document.getElementById("pollRecoveryBar");
  const msgEl = document.getElementById("pollRecoveryMsg");
  const btn = document.getElementById("btnResumePoll");
  if (!bar || !msgEl) return;
  bar.className = paused ? "notice n-err" : "notice n-warn";
  msgEl.textContent = message;
  if (btn) btn.textContent = paused ? "恢复状态轮询 / 重新同步当前任务" : "立即重试状态刷新";
  bar.style.display = "block";
}

function syncActionButtonsForSession(session) {
  if (!isLiveSessionCurrent(session)) return;
  const hasUpload = Boolean(LIVE_UPLOAD);
  const readOnly = Boolean(LIVE_UPLOAD && LIVE_UPLOAD.read_only_playback_only);
  const running = Boolean(session && session.isRunning && !session.terminalLocked);
  document.getElementById("btnStartCompare").disabled = !hasUpload || running || readOnly;
  document.getElementById("btnStartJob").disabled = !hasUpload || running || readOnly;
  document.getElementById("btnCancelJob").disabled = !running;
}

function beginLiveSession({
  kind = "idle",
  uploadId = null,
  uploadSha256 = null,
  comparisonId = null,
  jobId = null,
  isRunning = false,
} = {}) {
  const prev = LIVE_ACTIVE_SESSION;
  if (prev) {
    stopSessionTimer(prev);
    if (prev.abortController && typeof prev.abortController.abort === "function") {
      try { prev.abortController.abort(); } catch (_) {}
    }
  }
  LIVE_SESSION_SEQ += 1;
  LIVE_FOCUS_SEQ += 1;
  hidePollRecovery();
  const next = {
    sessionId: LIVE_SESSION_SEQ,
    kind,
    uploadId,
    uploadSha256,
    comparisonId,
    jobId,
    timerId: null,
    abortController: _makeAbortController(),
    pollFailures: 0,
    pollSeq: 0,
    appliedPollSeq: 0,
    inFlightCount: 0,
    pollInFlight: false,
    terminalLocked: !Boolean(isRunning),
    isRunning: Boolean(isRunning),
  };
  LIVE_ACTIVE_SESSION = next;
  return next;
}

function isLiveSessionCurrent(session, expected = {}) {
  if (!session || LIVE_ACTIVE_SESSION !== session || LIVE_ACTIVE_SESSION.sessionId !== session.sessionId) {
    return false;
  }
  if (expected.uploadId !== undefined && expected.uploadId !== null) {
    if (!LIVE_UPLOAD || LIVE_UPLOAD.upload_id !== expected.uploadId) return false;
  }
  if (expected.uploadSha256 !== undefined && expected.uploadSha256 !== null) {
    if (!LIVE_UPLOAD || LIVE_UPLOAD.source_image_sha256 !== expected.uploadSha256) return false;
  }
  if (expected.comparisonId !== undefined && expected.comparisonId !== null) {
    if (LIVE_COMPARISON_ID !== expected.comparisonId) return false;
  }
  if (expected.jobId !== undefined && expected.jobId !== null) {
    if (LIVE_JOB_ID !== expected.jobId) return false;
  }
  return true;
}

function startSessionPolling(session, pollFn, intervalMs = 800) {
  if (!isLiveSessionCurrent(session) || session.terminalLocked) return null;
  stopSessionTimer(session);
  let ownTid = null;
  ownTid = setInterval(() => {
    if (!isLiveSessionCurrent(session) || session.terminalLocked || session.pollInFlight) {
      return;
    }
    return pollFn(session, ownTid);
  }, intervalMs);
  session.timerId = ownTid;
  LIVE_POLL_TIMER = ownTid;
  return ownTid;
}

async function apiJson(url, opts) {
  const r = await fetch(url, opts);
  const d = await r.json();
  if (!r.ok) {
    const err = new Error(d.error || ("HTTP " + r.status));
    err.status = r.status;
    throw err;
  }
  return d;
}
function pct(v) { return (v === null || v === undefined) ? "N/A" : (v * 100).toFixed(2) + "%"; }
function activeClassSet() {
  const s = new Set();
  document.querySelectorAll(".chk-cls").forEach(el => { if (el.checked) s.add(el.value); });
  return s;
}
function outcomeBadgeClass(code) {
  if (code === "completed_with_boxes") return "b-ok";
  if (code === "completed_zero_boxes") return "b-zero";
  if (code === "failed" || code === "incomplete") return "b-rej";
  if (code === "running") return "b-run";
  return "b-wait";
}
function setModelLeftBadge(mk, text, bcls) {
  const el = document.getElementById("stBadge_" + mk);
  if (!el) return;
  el.className = "badge " + (bcls || "b-wait");
  el.textContent = text;
}
function resetAllModelLeftBadges() {
  for (const mk of ["qwen2", "deepseek", "qwen3"]) {
    setModelLeftBadge(mk, "未运行", "b-wait");
  }
}

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

// Model / mode synchronization
function syncModeOptions() {
  const mk = document.getElementById("liveModel").value;
  const selMode = document.getElementById("liveMode");
  const modeVal = selMode.value;
  const chkQ2 = document.getElementById("chkModel_qwen2");
  const chkDS = document.getElementById("chkModel_deepseek");
  const chkQ3 = document.getElementById("chkModel_qwen3");
  const addonWrap = document.getElementById("lblAddonBOther");
  const addonChk = document.getElementById("chkAddonQwen3BOther");

  if (modeVal === "T_B_other") {
    // T_B_other is strictly Qwen3-only enhancement
    document.getElementById("liveModel").value = "qwen3";
    chkQ2.checked = false; chkQ2.disabled = true;
    chkDS.checked = false; chkDS.disabled = true;
    chkQ3.checked = true;  chkQ3.disabled = false;
    addonChk.checked = false;
    addonWrap.style.display = "none";
    document.getElementById("modeHint").textContent =
      "当前选择冻结方法 Qwen3_T_plus_B_other：仅适用于 Qwen3-VL-2B-Instruct，先执行 1024px 切片与 NMS，再对每个候选框做 10% 外扩裁剪与二阶段过滤。";
  } else {
    chkQ2.disabled = false;
    chkDS.disabled = false;
    chkQ3.disabled = false;
    addonWrap.style.display = "flex";
    if (modeVal === "W") {
      document.getElementById("modeHint").textContent =
        "整图 ROI 模式（W · 默认同图对比基线）：每模型 1 次整板推理，无切片 NMS；三个模型在同一 ROI 输入上各自采用其冻结原生视觉预处理。";
    } else {
      const tc = LIVE_UPLOAD ? LIVE_UPLOAD.tile_count : "N";
      document.getElementById("modeHint").textContent =
        `1024px 切片模式（T）：每模型 ${tc} 次切片调用 + 0.50 IoU 跨切片 NMS 去重。`;
    }
  }
  const bOpt = selMode.querySelector('option[value="T_B_other"]');
  if (bOpt) bOpt.disabled = (mk !== "qwen3" && selMode.value !== "T_B_other" && !chkQ3.checked);
}
document.getElementById("liveModel").onchange = syncModeOptions;
document.getElementById("liveMode").onchange = syncModeOptions;

document.getElementById("btnSelAllModels").onclick = () => {
  if (document.getElementById("liveMode").value === "T_B_other") {
    document.getElementById("liveMode").value = "W";
    syncModeOptions();
  }
  for (const mk of ["qwen2", "deepseek", "qwen3"]) {
    const el = document.getElementById("chkModel_" + mk);
    if (!el.disabled) el.checked = true;
  }
};
document.getElementById("btnSelOnlyFocus").onclick = () => {
  const focusMk = document.getElementById("liveModel").value;
  for (const mk of ["qwen2", "deepseek", "qwen3"]) {
    const el = document.getElementById("chkModel_" + mk);
    el.checked = (mk === focusMk);
  }
};

// Class filter buttons
document.querySelectorAll(".chk-cls").forEach(el => {
  el.onchange = () => {
    renderComparisonPanel();
    renderLiveItemsTable();
    renderLiveCanvas();
  };
});
document.getElementById("btnClsAll").onclick = () => {
  document.querySelectorAll(".chk-cls").forEach(el => { el.checked = true; });
  renderComparisonPanel();
  renderLiveItemsTable();
  renderLiveCanvas();
};
document.getElementById("btnClsNone").onclick = () => {
  document.querySelectorAll(".chk-cls").forEach(el => { el.checked = false; });
  renderComparisonPanel();
  renderLiveItemsTable();
  renderLiveCanvas();
};
document.getElementById("liveCoordSpace").onchange = () => {
  renderComparisonPanel();
  renderLiveCanvas();
};

function clearLiveResultsForNewUpload(startFreshSession = true) {
  if (startFreshSession) {
    beginLiveSession({ kind: "idle", isRunning: false });
  } else {
    stopSessionTimer(LIVE_ACTIVE_SESSION);
    hidePollRecovery();
  }
  LIVE_JOB_ID = null;
  LIVE_RESULT = null;
  LIVE_COMPARISON_ID = null;
  LIVE_COMPARISON = null;
  LIVE_CROP = null;
  resetAllModelLeftBadges();
  document.getElementById("compareSectionCard").style.display = "none";
  document.getElementById("compareCardsGrid").innerHTML = "";
  document.getElementById("compareSummaryBody").innerHTML = "";
  document.getElementById("compareAddonWrap").style.display = "none";
  document.getElementById("compareAddonGrid").innerHTML = "";
  document.getElementById("focusModelSwitchBar").innerHTML = "";
  document.getElementById("liveResultCard").style.display = "none";
  document.getElementById("jobProgressCard").style.display = "none";
  document.getElementById("jobErrorBox").style.display = "none";
  document.getElementById("btnEvalGt").disabled = true;
  document.getElementById("btnCancelJob").disabled = true;
  document.getElementById("gtStatusBanner").className = "notice n-warn";
  document.getElementById("gtStatusBanner").innerHTML =
    "<strong>模型预测 / 未做精度评价（GT-free）</strong>：当前未提供人工标注真值，仅展示模型预测位置与类别计数；预测框数不代表板上真实器件数，不计算准确率/F1，<code>score=1.0</code> 仅为离散坐标占位符而非 100% 置信度。";
}

function renderUploadMetaBox(data) {
  const budgetNote = data.tile_count <= 24
    ? `<span class="badge b-ok">切片数 ${data.tile_count} &le; 24（支持快速运行 T + B_other）</span>`
    : `<span class="badge b-zero">切片数 ${data.tile_count} &gt; 24（现场推荐优先使用 W 整图三模型同图对比）</span>`;
  const readOnlyNotice = data.read_only_playback_only
    ? `<div class="notice n-warn" style="margin-top:6px;"><strong>只读回放样例（已禁用重新推理）</strong>：${data.playback_preview_notice || "当前记录为离线轻量包内置只读回放示例，派生预览图仅供历史结果检视、筛选、缩放与导出；如需发起新推理请先上传本地原图。"}</div>`
    : "";
  document.getElementById("uploadMetaBox").innerHTML =
    `<div>已通过 GT-free 图像预检：<strong>${data.safe_filename}</strong></div>` +
    `<div>原图实际尺寸：<strong>${data.orig_wh[0]} &times; ${data.orig_wh[1]}</strong> px | 裁切后 ROI：<strong>${data.roi_width} &times; ${data.roi_height}</strong> px（偏移 [${data.roi_offset_x}, ${data.roi_offset_y}]）</div>` +
    `<div>可用模式与工作量：<strong>W</strong> = 1 次调用/模型（三模型共 3 次） | <strong>T</strong> = ${data.tile_count} 次切片调用/模型 | ${budgetNote}</div>` +
    readOnlyNotice +
    `<details style="margin-top:4px;"><summary style="cursor:pointer;">展开图像身份与校验详情 (upload_id / ROI 边界 / SHA-256)</summary>` +
    `<div style="margin-top:4px;">upload_id: <code>${data.upload_id}</code> | ROI half-open: [${(data.roi_bbox_half_open || []).join(", ")}]<br/>原图 SHA-256: <code>${data.source_image_sha256}</code></div></details>`;
}

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
  const uploadSeq = ++LIVE_UPLOAD_SEQ;
  const sessionBefore = LIVE_ACTIVE_SESSION.sessionId;
  document.getElementById("btnUpload").disabled = true;
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(f.name) },
      body: f,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
    if (uploadSeq !== LIVE_UPLOAD_SEQ || LIVE_ACTIVE_SESSION.sessionId !== sessionBefore) {
      return;
    }
    clearLiveResultsForNewUpload(false);
    LIVE_UPLOAD = data;
    const session = beginLiveSession({
      kind: "upload",
      uploadId: data.upload_id,
      uploadSha256: data.source_image_sha256,
      isRunning: false,
    });
    syncActionButtonsForSession(session);
    renderUploadMetaBox(data);
    syncModeOptions();
    renderLiveCanvas();
  } catch (err) {
    if (uploadSeq !== LIVE_UPLOAD_SEQ) return;
    msgEl.className = "notice n-err";
    msgEl.textContent = "上传或预检未通过：" + err.message;
    msgEl.style.display = "block";
  } finally {
    if (uploadSeq === LIVE_UPLOAD_SEQ) {
      document.getElementById("btnUpload").disabled = false;
    }
  }
};

// Step 2A: Start Multi-Model Serial Comparison Queue
document.getElementById("btnStartCompare").onclick = async () => {
  if (!LIVE_UPLOAD) return;
  const errBox = document.getElementById("jobErrorBox");
  errBox.style.display = "none";
  if (LIVE_UPLOAD.read_only_playback_only) {
    errBox.textContent = LIVE_UPLOAD.playback_preview_notice || "当前上传记录为只读回放示例，禁止重新发起推理；请上传新图后再运行。";
    errBox.style.display = "block";
    syncActionButtonsForSession(LIVE_ACTIVE_SESSION);
    return;
  }
  const mode = document.getElementById("liveMode").value;
  const models = [];
  for (const mk of ["qwen2", "deepseek", "qwen3"]) {
    const chk = document.getElementById("chkModel_" + mk);
    if (chk && chk.checked && !chk.disabled) models.push(mk);
  }
  if (!models.length) {
    errBox.textContent = "请至少勾选一个参测模型。";
    errBox.style.display = "block";
    return;
  }
  const includeAddon = Boolean(document.getElementById("chkAddonQwen3BOther").checked && mode !== "T_B_other");
  const targetUpload = LIVE_UPLOAD;
  const reqSession = beginLiveSession({
    kind: "comparison",
    uploadId: targetUpload.upload_id,
    uploadSha256: targetUpload.source_image_sha256,
    isRunning: true,
  });
  syncActionButtonsForSession(reqSession);
  try {
    const fetchOpts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        upload_id: targetUpload.upload_id,
        model_keys: models,
        mode: mode,
        include_qwen3_b_other: includeAddon,
      }),
    };
    if (reqSession.abortController) fetchOpts.signal = reqSession.abortController.signal;
    const cmp = await apiJson("/api/comparisons", fetchOpts);
    if (!isLiveSessionCurrent(reqSession, { uploadId: targetUpload.upload_id, uploadSha256: targetUpload.source_image_sha256 })) {
      return;
    }
    if (cmp.upload_id !== targetUpload.upload_id || cmp.source_image_sha256 !== targetUpload.source_image_sha256) {
      return;
    }
    reqSession.comparisonId = cmp.comparison_id;
    const isRunning = !isComparisonTerminalStatus(cmp.status);
    reqSession.isRunning = isRunning;
    reqSession.terminalLocked = !isRunning;
    LIVE_COMPARISON_ID = cmp.comparison_id;
    LIVE_COMPARISON = cmp;
    LIVE_JOB_ID = null;
    LIVE_RESULT = null;
    LIVE_CROP = null;
    document.getElementById("liveResultCard").style.display = "none";
    document.getElementById("btnEvalGt").disabled = true;
    document.getElementById("jobProgressCard").style.display = "block";
    syncActionButtonsForSession(reqSession);
    renderComparisonPanel();
    updateComparisonProgressUi(cmp);
    if (reqSession.isRunning) {
      startSessionPolling(reqSession, pollComparison, 800);
    }
  } catch (err) {
    if (!isLiveSessionCurrent(reqSession)) return;
    reqSession.isRunning = false;
    reqSession.terminalLocked = true;
    syncActionButtonsForSession(reqSession);
    errBox.textContent = err.message;
    errBox.style.display = "block";
  }
};

// Step 2B: Start Single Focused Model Job (backward-compatible single job entry)
document.getElementById("btnStartJob").onclick = async () => {
  if (!LIVE_UPLOAD) return;
  const errBox = document.getElementById("jobErrorBox");
  errBox.style.display = "none";
  if (LIVE_UPLOAD.read_only_playback_only) {
    errBox.textContent = LIVE_UPLOAD.playback_preview_notice || "当前上传记录为只读回放示例，禁止重新发起推理；请上传新图后再运行。";
    errBox.style.display = "block";
    syncActionButtonsForSession(LIVE_ACTIVE_SESSION);
    return;
  }
  const mk = document.getElementById("liveModel").value;
  const mode = document.getElementById("liveMode").value;
  const targetUpload = LIVE_UPLOAD;
  const reqSession = beginLiveSession({
    kind: "job",
    uploadId: targetUpload.upload_id,
    uploadSha256: targetUpload.source_image_sha256,
    isRunning: true,
  });
  // Clear prior comparison binding immediately when switching to single-model job on the same upload
  LIVE_COMPARISON_ID = null;
  LIVE_COMPARISON = null;
  LIVE_JOB_ID = null;
  LIVE_RESULT = null;
  LIVE_CROP = null;
  resetAllModelLeftBadges();
  document.getElementById("compareSectionCard").style.display = "none";
  document.getElementById("liveResultCard").style.display = "none";
  document.getElementById("btnEvalGt").disabled = true;
  syncActionButtonsForSession(reqSession);
  try {
    const fetchOpts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: targetUpload.upload_id, model_key: mk, mode: mode }),
    };
    if (reqSession.abortController) fetchOpts.signal = reqSession.abortController.signal;
    const job = await apiJson("/api/jobs", fetchOpts);
    if (!isLiveSessionCurrent(reqSession, { uploadId: targetUpload.upload_id, uploadSha256: targetUpload.source_image_sha256 })) {
      return;
    }
    if (job.upload_id !== targetUpload.upload_id) return;
    reqSession.jobId = job.job_id;
    const isRunning = !isJobTerminalStatus(job.status);
    reqSession.isRunning = isRunning;
    reqSession.terminalLocked = !isRunning;
    LIVE_JOB_ID = job.job_id;
    document.getElementById("jobProgressCard").style.display = "block";
    setModelLeftBadge(mk, "运行中", "b-run");
    updateProgressUi(job);
    syncActionButtonsForSession(reqSession);
    if (reqSession.isRunning) {
      startSessionPolling(reqSession, pollJob, 800);
    }
  } catch (err) {
    if (!isLiveSessionCurrent(reqSession)) return;
    reqSession.isRunning = false;
    reqSession.terminalLocked = true;
    syncActionButtonsForSession(reqSession);
    errBox.textContent = err.message;
    errBox.style.display = "block";
  }
};

document.getElementById("btnCancelJob").onclick = async () => {
  const session = LIVE_ACTIVE_SESSION;
  if (!session || !session.isRunning || session.terminalLocked) return;
  const targetKind = session.kind;
  const targetCmpId = (targetKind === "comparison") ? (session.comparisonId || LIVE_COMPARISON_ID) : null;
  const targetJobId = (targetKind === "job") ? (session.jobId || LIVE_JOB_ID) : null;
  if (!targetCmpId && !targetJobId) return;

  const cancelSeq = ++session.pollSeq;
  document.getElementById("btnCancelJob").disabled = true;
  try {
    if (targetKind === "comparison" && targetCmpId) {
      const cmp = await apiJson(`/api/comparisons/${encodeURIComponent(targetCmpId)}/cancel`, { method: "POST" });
      if (!isLiveSessionCurrent(session, { comparisonId: targetCmpId })) return;
      if (cmp.comparison_id !== targetCmpId) return;
      if (session.terminalLocked && cancelSeq < (session.appliedPollSeq || 0)) return;
      session.appliedPollSeq = Math.max(session.appliedPollSeq || 0, cancelSeq);
      if (isComparisonTerminalStatus(cmp.status)) {
        session.terminalLocked = true;
        session.isRunning = false;
        stopSessionTimer(session);
        hidePollRecovery();
      }
      LIVE_COMPARISON = cmp;
      updateComparisonProgressUi(cmp);
      renderComparisonPanel();
      syncActionButtonsForSession(session);
    } else if (targetJobId) {
      const st = await apiJson(`/api/jobs/${encodeURIComponent(targetJobId)}/cancel`, { method: "POST" });
      if (!isLiveSessionCurrent(session, { jobId: targetJobId })) return;
      if (st.job_id !== targetJobId) return;
      if (session.terminalLocked && cancelSeq < (session.appliedPollSeq || 0)) return;
      session.appliedPollSeq = Math.max(session.appliedPollSeq || 0, cancelSeq);
      if (isJobTerminalStatus(st.status)) {
        session.terminalLocked = true;
        session.isRunning = false;
        stopSessionTimer(session);
        hidePollRecovery();
      }
      updateProgressUi(st);
      if (st.outcome) {
        setModelLeftBadge(st.model_key, st.outcome.outcome_label, outcomeBadgeClass(st.outcome.outcome_code));
      }
      syncActionButtonsForSession(session);
    }
  } catch (err) {
    if (!isLiveSessionCurrent(session) || session.terminalLocked) return;
    document.getElementById("jobErrorBox").textContent = "取消请求暂时失败（可再次点击取消重试）：" + err.message;
    document.getElementById("jobErrorBox").style.display = "block";
    if (session.isRunning && !session.terminalLocked) {
      document.getElementById("btnCancelJob").disabled = false;
    }
  }
};

document.getElementById("btnResumePoll").onclick = () => {
  const session = LIVE_ACTIVE_SESSION;
  if (!session || session.terminalLocked) return;
  session.pollFailures = 0;
  hidePollRecovery();
  if (session.kind === "comparison" && session.comparisonId) {
    if (!session.timerId) startSessionPolling(session, pollComparison, 800);
    if (!session.pollInFlight) pollComparison(session, session.timerId);
  } else if (session.kind === "job" && session.jobId) {
    if (!session.timerId) startSessionPolling(session, pollJob, 800);
    if (!session.pollInFlight) pollJob(session, session.timerId);
  }
};

function updateProgressUi(st) {
  document.getElementById("jobStageLbl").textContent = `[${st.model_display_name || st.model_key} · ${st.status}] ${st.stage_label}`;
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

function updateComparisonProgressUi(cmp) {
  const slots = cmp.slots || [];
  const totalSlots = slots.length;
  let doneSlots = 0;
  let activeSlot = null;
  for (const s of slots) {
    const out = s.outcome || {};
    if (s.slot_role !== "separate_qwen3_enhancement") {
      setModelLeftBadge(s.model_key, out.outcome_label || s.queue_status, outcomeBadgeClass(out.outcome_code));
    }
    if (["completed", "failed", "cancelled", "incomplete"].includes(s.queue_status)) {
      doneSlots++;
    } else if (["running", "running_base_inference", "preparing_verifier", "running_verifier", "queued"].includes(s.queue_status) && !activeSlot) {
      activeSlot = s;
    }
  }
  const pctDone = totalSlots > 0 ? Math.round((doneSlots / totalSlots) * 100) : 100;
  document.getElementById("jobProgressFill").style.width = (cmp.status !== "running" ? 100 : Math.max(8, pctDone)) + "%";
  if (activeSlot) {
    document.getElementById("jobStageLbl").textContent =
      `[串行队列 ${doneSlots + 1}/${totalSlots}] ${activeSlot.model_display_name} (${activeSlot.mode}) — ${activeSlot.stage_label || activeSlot.queue_status}`;
  } else {
    document.getElementById("jobStageLbl").textContent =
      `[同图对比队列 ${cmp.status}] 已完成 ${doneSlots}/${totalSlots} 个模型任务`;
  }
  document.getElementById("jobTimeLbl").textContent = `队列耗时 ${cmp.wall_elapsed_seconds}s（累计推理 ${cmp.sum_job_elapsed_seconds}s）`;
  const summaryParts = slots.map(s => `${s.model_display_name}[${s.mode}]: ${(s.outcome && s.outcome.outcome_label) || s.queue_status}`);
  document.getElementById("jobCountsLbl").textContent = summaryParts.join("  |  ");
}

async function pollComparison(boundSession = null, ownTimerId = null) {
  const session = boundSession || LIVE_ACTIVE_SESSION;
  if (!session || session.terminalLocked) return;
  const targetCmpId = (boundSession && boundSession.comparisonId) || LIVE_COMPARISON_ID;
  const targetUploadId = (boundSession && boundSession.uploadId) || (LIVE_UPLOAD && LIVE_UPLOAD.upload_id) || null;
  const targetUploadSha = (boundSession && boundSession.uploadSha256) || (LIVE_UPLOAD && LIVE_UPLOAD.source_image_sha256) || null;
  const timerAtStart = (ownTimerId !== null && ownTimerId !== undefined)
    ? ownTimerId
    : (session ? session.timerId : LIVE_POLL_TIMER);
  if (!targetCmpId) return;
  const reqSeq = ++session.pollSeq;
  session.inFlightCount = (session.inFlightCount || 0) + 1;
  session.pollInFlight = true;
  try {
    const fetchOpts = (session && session.abortController) ? { signal: session.abortController.signal } : undefined;
    const cmp = await apiJson(`/api/comparisons/${encodeURIComponent(targetCmpId)}`, fetchOpts);
    // Strict post-await session + upload + comparison_id + terminal lock + monotonic sequence guard:
    if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha, comparisonId: targetCmpId })) {
      return;
    }
    if (session.terminalLocked || reqSeq < (session.appliedPollSeq || 0)) {
      return;
    }
    if (
      cmp.comparison_id !== targetCmpId ||
      (LIVE_UPLOAD && (cmp.upload_id !== LIVE_UPLOAD.upload_id || cmp.source_image_sha256 !== LIVE_UPLOAD.source_image_sha256))
    ) {
      return;
    }
    if (timerAtStart !== null && timerAtStart !== undefined && session.timerId !== null && session.timerId !== timerAtStart) {
      return;
    }

    session.appliedPollSeq = reqSeq;
    session.pollFailures = 0;
    hidePollRecovery();
    const reachedTerminal = isComparisonTerminalStatus(cmp.status);
    if (reachedTerminal) {
      session.terminalLocked = true;
      session.isRunning = false;
      stopSessionTimer(session, timerAtStart);
    }
    LIVE_COMPARISON = cmp;
    updateComparisonProgressUi(cmp);
    renderComparisonPanel();
    if (reachedTerminal) {
      syncActionButtonsForSession(session);
    }

    // Auto-focus the latest completed slot in the main inspection canvas if none selected for THIS comparison yet
    const completedSlots = (cmp.slots || []).filter(s => s.queue_status === "completed" && s.has_result && s.job_id);
    const slotIds = new Set(completedSlots.map(s => s.job_id));
    if (completedSlots.length > 0 && (!LIVE_RESULT || LIVE_RESULT.upload_id !== cmp.upload_id || !slotIds.has(LIVE_RESULT.job_id))) {
      const pick = completedSlots.find(s => s.kept_predictions_count > 0) || completedSlots[0];
      await focusComparisonSlot(pick.job_id, session);
      if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha, comparisonId: targetCmpId })) {
        return;
      }
    }

    if (reachedTerminal) {
      refreshRecentJobs();
      refreshRecentComparisons();
    }
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha, comparisonId: targetCmpId })) {
      return;
    }
    if (session.terminalLocked || reqSeq < (session.appliedPollSeq || 0)) {
      return;
    }
    session.pollFailures = (session.pollFailures || 0) + 1;
    const isFatal4xx = Boolean(err && err.status >= 400 && err.status < 500);
    if (isFatal4xx || session.pollFailures >= 3) {
      stopSessionTimer(session, timerAtStart);
      showPollRecovery(
        session,
        `对比队列状态刷新暂时失败（${err.message || "网络异常"}），已暂停自动轮询；当前任务标识与取消按钮保持绑定，可点击下方按钮恢复刷新。`,
        true
      );
      if (isFatal4xx) {
        session.isRunning = false;
        syncActionButtonsForSession(session);
      }
    } else {
      showPollRecovery(
        session,
        `状态轮询请求暂时失败（第 ${session.pollFailures} 次：${err.message || "网络抖动"}），将继续自动重试或点击立即刷新。`,
        false
      );
    }
  } finally {
    session.inFlightCount = Math.max(0, (session.inFlightCount || 1) - 1);
    session.pollInFlight = session.inFlightCount > 0;
  }
}

async function pollJob(boundSession = null, ownTimerId = null) {
  const session = boundSession || LIVE_ACTIVE_SESSION;
  if (!session || session.terminalLocked) return;
  const targetJobId = (boundSession && boundSession.jobId) || LIVE_JOB_ID;
  const targetUploadId = (boundSession && boundSession.uploadId) || (LIVE_UPLOAD && LIVE_UPLOAD.upload_id) || null;
  const targetUploadSha = (boundSession && boundSession.uploadSha256) || (LIVE_UPLOAD && LIVE_UPLOAD.source_image_sha256) || null;
  const timerAtStart = (ownTimerId !== null && ownTimerId !== undefined)
    ? ownTimerId
    : (session ? session.timerId : LIVE_POLL_TIMER);
  if (!targetJobId) return;
  const reqSeq = ++session.pollSeq;
  session.inFlightCount = (session.inFlightCount || 0) + 1;
  session.pollInFlight = true;
  try {
    const fetchOpts = (session && session.abortController) ? { signal: session.abortController.signal } : undefined;
    const st = await apiJson(`/api/jobs/${encodeURIComponent(targetJobId)}`, fetchOpts);
    if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha, jobId: targetJobId })) {
      return;
    }
    if (session.terminalLocked || reqSeq < (session.appliedPollSeq || 0)) {
      return;
    }
    if (
      st.job_id !== targetJobId ||
      (LIVE_UPLOAD && st.upload_id !== LIVE_UPLOAD.upload_id)
    ) {
      return;
    }
    if (timerAtStart !== null && timerAtStart !== undefined && session.timerId !== null && session.timerId !== timerAtStart) {
      return;
    }

    session.appliedPollSeq = reqSeq;
    session.pollFailures = 0;
    hidePollRecovery();
    const reachedTerminal = isJobTerminalStatus(st.status);
    if (reachedTerminal) {
      session.terminalLocked = true;
      session.isRunning = false;
      stopSessionTimer(session, timerAtStart);
    }
    updateProgressUi(st);
    if (st.outcome) {
      setModelLeftBadge(st.model_key, st.outcome.outcome_label, outcomeBadgeClass(st.outcome.outcome_code));
    }
    if (reachedTerminal) {
      syncActionButtonsForSession(session);
      refreshRecentJobs();
      if (st.status === "completed") {
        const res = await apiJson(`/api/jobs/${encodeURIComponent(targetJobId)}/result`, fetchOpts);
        if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha, jobId: targetJobId })) {
          return;
        }
        if (res.job_id !== targetJobId || (LIVE_UPLOAD && res.upload_id !== LIVE_UPLOAD.upload_id)) return;
        LIVE_RESULT = res;
        document.getElementById("btnEvalGt").disabled = false;
        renderLiveResult();
      } else {
        const eb = document.getElementById("jobErrorBox");
        eb.textContent = st.error_message || ("任务终止状态：" + st.status);
        eb.style.display = "block";
      }
    }
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha, jobId: targetJobId })) {
      return;
    }
    if (session.terminalLocked || reqSeq < (session.appliedPollSeq || 0)) {
      return;
    }
    session.pollFailures = (session.pollFailures || 0) + 1;
    const isFatal4xx = Boolean(err && err.status >= 400 && err.status < 500);
    if (isFatal4xx || session.pollFailures >= 3) {
      stopSessionTimer(session, timerAtStart);
      showPollRecovery(
        session,
        `单任务状态刷新暂时失败（${err.message || "网络异常"}），已暂停自动轮询；可点击下方按钮恢复状态同步。`,
        true
      );
      if (isFatal4xx) {
        session.isRunning = false;
        syncActionButtonsForSession(session);
      }
    } else {
      showPollRecovery(
        session,
        `单任务状态轮询暂时失败（第 ${session.pollFailures} 次：${err.message || "网络抖动"}），将继续自动重试或点击立即刷新。`,
        false
      );
    }
  } finally {
    session.inFlightCount = Math.max(0, (session.inFlightCount || 1) - 1);
    session.pollInFlight = session.inFlightCount > 0;
  }
}

async function focusComparisonSlot(jobId, boundSession = null) {
  if (!jobId) return;
  const session = boundSession || LIVE_ACTIVE_SESSION;
  const focusSeq = ++LIVE_FOCUS_SEQ;
  const expectedUploadId = LIVE_UPLOAD ? LIVE_UPLOAD.upload_id : null;
  const expectedUploadSha = LIVE_UPLOAD ? LIVE_UPLOAD.source_image_sha256 : null;
  const expectedCmpId = LIVE_COMPARISON_ID;
  try {
    const fetchOpts = (session && session.abortController) ? { signal: session.abortController.signal } : undefined;
    const res = await apiJson(`/api/jobs/${encodeURIComponent(jobId)}/result`, fetchOpts);
    if (
      focusSeq !== LIVE_FOCUS_SEQ ||
      !isLiveSessionCurrent(session, { uploadId: expectedUploadId, uploadSha256: expectedUploadSha }) ||
      LIVE_COMPARISON_ID !== expectedCmpId
    ) {
      return;
    }
    if (LIVE_UPLOAD && (res.upload_id !== LIVE_UPLOAD.upload_id || res.source_image_sha256 !== LIVE_UPLOAD.source_image_sha256)) {
      throw new Error("拒绝跨图加载：该任务不属于当前上传图像。");
    }
    LIVE_JOB_ID = jobId;
    LIVE_RESULT = res;
    document.getElementById("liveModel").value = res.model_key;
    document.getElementById("btnEvalGt").disabled = false;
    renderLiveResult();
    renderComparisonPanel();
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (focusSeq !== LIVE_FOCUS_SEQ || !isLiveSessionCurrent(session)) return;
    const eb = document.getElementById("jobErrorBox");
    eb.textContent = "加载模型结果失败：" + err.message;
    eb.style.display = "block";
  }
}

function buildOverlaySvgMarkup(predictions, preCandidates, gts, roi, origWh, coordSpace, cropRoi, showView) {
  const clsSet = activeClassSet();
  const rox = roi.offset_x, roy = roi.offset_y, roiW = roi.width, roiH = roi.height;
  const origW = origWh ? origWh[0] : (roiW + rox);
  const origH = origWh ? origWh[1] : (roiH + roy);

  let vx1, vy1, vx2, vy2;
  if (coordSpace === "orig") {
    if (cropRoi) {
      vx1 = cropRoi[0] + rox; vy1 = cropRoi[1] + roy;
      vx2 = cropRoi[2] + rox; vy2 = cropRoi[3] + roy;
    } else {
      vx1 = 0; vy1 = 0; vx2 = origW; vy2 = origH;
    }
  } else {
    vx1 = cropRoi ? cropRoi[0] : 0;
    vy1 = cropRoi ? cropRoi[1] : 0;
    vx2 = cropRoi ? cropRoi[2] : roiW;
    vy2 = cropRoi ? cropRoi[3] : roiH;
  }
  const vw = Math.max(1, vx2 - vx1), vh = Math.max(1, vy2 - vy1);
  const sw = Math.max(2, vw / 420);
  const fs = Math.max(12, Math.round(vw / 72));
  let elems = "";

  if (coordSpace === "orig") {
    elems += `<rect x="${rox}" y="${roy}" width="${roiW}" height="${roiH}" fill="none" stroke="#94a3b8" stroke-width="${sw}" stroke-dasharray="${sw * 4} ${sw * 3}" />`;
    elems += `<text x="${rox + 6}" y="${Math.max(vy1 + fs, roy + fs + 4)}" fill="#94a3b8" font-size="${fs}" font-weight="bold" style="paint-order:stroke;stroke:#020617;stroke-width:${Math.max(3, fs / 4)}px;">ROI [${rox},${roy},${rox + roiW},${roy + roiH}]</text>`;
  }

  const drawBox = (b, color, dash, text) => {
    const x1 = coordSpace === "orig" ? b[0] : (b[0] - rox);
    const y1 = coordSpace === "orig" ? b[1] : (b[1] - roy);
    const x2 = coordSpace === "orig" ? b[2] : (b[2] - rox);
    const y2 = coordSpace === "orig" ? b[3] : (b[3] - roy);
    const w = Math.max(2, x2 - x1), h = Math.max(2, y2 - y1);
    elems += `<rect x="${x1}" y="${y1}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="${sw}" ${dash ? 'stroke-dasharray="' + (sw * 3) + ' ' + (sw * 2) + '"' : ""} />`;
    elems += `<text x="${x1}" y="${Math.max(vy1 + fs, y1 - 4)}" fill="${color}" font-size="${fs}" font-weight="bold" style="paint-order:stroke;stroke:#020617;stroke-width:${Math.max(3, fs / 4)}px;">${text}</text>`;
  };

  if (gts) {
    for (const g of gts) {
      if (g.match_status === "FN" && clsSet.has(g.label)) {
        drawBox(g.bbox_xyxy, "#facc15", true, `FN:${CLASS_CN[g.label] || g.label}`);
      }
    }
  }

  let renderItems = [];
  if (showView === "pre_verifier" && Array.isArray(preCandidates)) {
    renderItems = preCandidates.map(c => ({ ...c, _rejected: !c.kept }));
  } else {
    renderItems = (predictions || []).map(p => ({ ...p, _rejected: false }));
    if (showView === "with_rejected" && Array.isArray(preCandidates)) {
      for (const c of preCandidates) {
        if (!c.kept) renderItems.push({ ...c, _rejected: true });
      }
    }
  }

  let visibleCount = 0;
  for (const item of renderItems) {
    if (!clsSet.has(item.label)) continue;
    visibleCount++;
    const col = item._rejected ? "#fb7185"
      : item.match_status === "FP" ? "#ef4444"
      : (CLASS_COLORS[item.label] || "#38bdf8");
    const cn = CLASS_CN[item.label] || item.label;
    const tag = item._rejected ? `剔除:${cn}` : `${cn}(${item.label})`;
    drawBox(item.bbox_xyxy, col, Boolean(item._rejected || item.match_status === "FP"), tag);
  }
  return { viewBox: `${vx1} ${vy1} ${vw} ${vh}`, vw, vh, svgHtml: elems, visibleCount };
}

function renderComparisonPanel() {
  const card = document.getElementById("compareSectionCard");
  if (!LIVE_COMPARISON || !LIVE_UPLOAD) {
    card.style.display = "none";
    return;
  }
  if (LIVE_COMPARISON.upload_id !== LIVE_UPLOAD.upload_id || LIVE_COMPARISON.source_image_sha256 !== LIVE_UPLOAD.source_image_sha256) {
    card.style.display = "none";
    return;
  }
  card.style.display = "block";
  const cmp = LIVE_COMPARISON;
  const roi = cmp.roi;
  const coordSpace = document.getElementById("liveCoordSpace").value;
  const clsSet = activeClassSet();

  document.getElementById("cmpIdentityBanner").innerHTML =
    `当前对比：<code>${cmp.comparison_id}</code> | 绑定图像：<strong>${cmp.safe_filename}</strong> | upload_id=<code>${cmp.upload_id}</code> | ` +
    `SHA-256=<code>${cmp.source_image_sha256.slice(0, 16)}…</code> | 原图=${cmp.orig_wh[0]}&times;${cmp.orig_wh[1]} | ` +
    `ROI=${roi.width}&times;${roi.height} (偏移 [${roi.offset_x}, ${roi.offset_y}]) | 统一对比基线模式=<strong>${cmp.baseline_mode_display_name}</strong>`;

  const tbody = document.getElementById("compareSummaryBody");
  tbody.innerHTML = "";
  const baseGrid = document.getElementById("compareCardsGrid");
  baseGrid.innerHTML = "";
  const addonWrap = document.getElementById("compareAddonWrap");
  const addonGrid = document.getElementById("compareAddonGrid");
  addonGrid.innerHTML = "";
  const focusBar = document.getElementById("focusModelSwitchBar");
  focusBar.innerHTML = "";

  const baselineSlots = (cmp.slots || []).filter(s => s.slot_role !== "separate_qwen3_enhancement");
  const addonSlots = (cmp.slots || []).filter(s => s.slot_role === "separate_qwen3_enhancement");
  addonWrap.style.display = addonSlots.length ? "block" : "none";

  let cropParam = "";
  if (LIVE_CROP) {
    if (coordSpace === "orig") {
      cropParam = `&crop=${LIVE_CROP[0] + roi.offset_x},${LIVE_CROP[1] + roi.offset_y},${LIVE_CROP[2] + roi.offset_x},${LIVE_CROP[3] + roi.offset_y}`;
    } else {
      cropParam = `&crop=${LIVE_CROP[0]},${LIVE_CROP[1]},${LIVE_CROP[2]},${LIVE_CROP[3]}`;
    }
  }
  const previewUrl = `/api/upload_preview/${encodeURIComponent(cmp.upload_id)}?source=${encodeURIComponent(coordSpace)}&max_width=${LIVE_CROP ? 1000 : 1200}${cropParam}`;

  for (const s of (cmp.slots || [])) {
    const out = s.outcome || {};
    const bcls = outcomeBadgeClass(out.outcome_code);
    const isCompleted = (s.queue_status === "completed" && s.has_result);
    const preds = isCompleted ? (s.predictions || []) : [];
    const visPreds = preds.filter(p => clsSet.has(p.label));
    const cc = isCompleted ? (s.class_counts || {}) : null;
    const tv2 = isCompleted ? (s.termination_v2 || {}) : null;

    // 1. Summary Table Row
    const tr = document.createElement("tr");
    const roleTag = s.slot_role === "separate_qwen3_enhancement" ? ` <span class="badge" style="background:rgba(192,132,252,0.22);color:#e9d5ff;">单独增强</span>` : "";
    if (isCompleted) {
      const termStr = `EOS=${tv2.normal_eos}/${tv2.total_calls}, 限长=${tv2.length_limited_without_eos}, 显式[]=${tv2.explicit_empty}`;
      tr.innerHTML =
        `<td><strong>${s.model_display_name}</strong>${roleTag}</td>` +
        `<td><code>${s.mode}</code></td>` +
        `<td><span class="badge ${bcls}">${out.outcome_label}</span></td>` +
        `<td>${s.elapsed_seconds}s</td>` +
        `<td><strong>${s.kept_predictions_count}</strong> (${visPreds.length})</td>` +
        `<td>${cc.capacitor || 0}</td><td>${cc.resistor || 0}</td><td>${cc.inductor || 0}</td>` +
        `<td>${cc.transistor || 0}</td><td>${cc.diode || 0}</td><td>${cc.ic || 0}</td>` +
        `<td style="font-size:11px;color:var(--muted);">${termStr}</td>` +
        `<td><button type="button" data-focus-job="${s.job_id}" style="padding:3px 7px;font-size:11px;">聚焦检视</button></td>`;
    } else {
      const noteTxt = s.error_message || out.outcome_note || "未完成有效推理（不生成空成绩）";
      tr.innerHTML =
        `<td><strong>${s.model_display_name}</strong>${roleTag}</td>` +
        `<td><code>${s.mode}</code></td>` +
        `<td><span class="badge ${bcls}">${out.outcome_label || s.queue_status}</span></td>` +
        `<td>${s.elapsed_seconds || 0}s</td>` +
        `<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>` +
        `<td style="font-size:11px;color:var(--muted);">${noteTxt}</td>` +
        `<td>—</td>`;
    }
    tbody.appendChild(tr);

    // 2. Side-by-Side Visual Card
    const col = document.createElement("div");
    col.className = "cmp-col" + (LIVE_JOB_ID && s.job_id === LIVE_JOB_ID ? " active-focus" : "");
    const overlayData = buildOverlaySvgMarkup(
      preds,
      s.pre_verifier_candidates,
      s.post_hoc_eval ? s.post_hoc_eval.gts : null,
      roi,
      cmp.orig_wh,
      coordSpace,
      LIVE_CROP,
      "kept"
    );

    let clsMiniHtml = "";
    if (isCompleted) {
      clsMiniHtml = `<div class="cls-mini-grid">` +
        CLASSES.map(c => `<div class="cls-mini-cell"><span><span class="cls-swatch" style="background:${CLASS_COLORS[c]};width:8px;height:8px;margin-right:4px;"></span>${CLASS_CN[c]}</span><strong>${cc[c] || 0}</strong></div>`).join("") +
        `</div>`;
    } else {
      clsMiniHtml = `<div class="notice ${out.outcome_code === 'failed' ? 'n-err' : 'n-warn'}" style="margin:0;padding:6px 8px;font-size:11.5px;">${s.error_message || out.outcome_note || '尚未生成预测结果，不展示空计数表。'}</div>`;
    }

    let statusNoteHtml = "";
    if (isCompleted && s.kept_predictions_count === 0) {
      statusNoteHtml = `<div class="notice n-warn" style="margin:0;padding:6px 8px;font-size:11.5px;">${out.outcome_note}</div>`;
    } else if (isCompleted) {
      statusNoteHtml = `<div style="font-size:11px;color:var(--muted);">${out.outcome_note}</div>`;
    }

    const expBtnsHtml = isCompleted
      ? `<div class="row" style="gap:5px;margin-top:auto;padding-top:4px;">` +
        `<button type="button" class="primary" data-focus-job="${s.job_id}" style="padding:4px 8px;font-size:11.5px;">设为下方主检视画布</button>` +
        `<a href="${s.exports.json}" style="font-size:11px;color:#38bdf8;text-decoration:none;padding:4px 6px;border:1px solid var(--line);border-radius:5px;">JSON</a>` +
        `<a href="${s.exports.overlay_png}" style="font-size:11px;color:#38bdf8;text-decoration:none;padding:4px 6px;border:1px solid var(--line);border-radius:5px;">PNG</a>` +
        `<a href="${s.exports.summary_md}" style="font-size:11px;color:#38bdf8;text-decoration:none;padding:4px 6px;border:1px solid var(--line);border-radius:5px;">摘要MD</a>` +
        `</div>`
      : `<div style="font-size:11px;color:var(--muted);margin-top:auto;">待模型完成有效推理后解锁聚焦检视与导出</div>`;

    col.innerHTML =
      `<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:6px;">` +
        `<div>` +
          `<div style="font-size:13px;font-weight:700;color:#e0f2fe;">${s.model_display_name}</div>` +
          `<div style="font-size:11px;color:var(--muted);">模式：<code>${s.mode}</code> | ${s.visual_preprocess_note || ""}</div>` +
        `</div>` +
        `<span class="badge ${bcls}">${out.outcome_label || s.queue_status}</span>` +
      `</div>` +
      `<div class="cmp-canvas-box">` +
        `<div class="stage" style="width:100%;">` +
          `<img src="${previewUrl}" alt="${s.model_display_name}" style="width:100%;height:auto;display:block;" />` +
          `<svg viewBox="${overlayData.viewBox}" preserveAspectRatio="none" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;">${overlayData.svgHtml}</svg>` +
        `</div>` +
      `</div>` +
      `<div style="display:flex;justify-content:space-between;font-size:11.5px;color:#cbd5e1;">` +
        `<span>预测框：<strong>${isCompleted ? s.kept_predictions_count : "—"}</strong>${isCompleted ? ` (筛选可见 ${overlayData.visibleCount})` : ""}</span>` +
        `<span>耗时：<strong>${s.elapsed_seconds || 0}s</strong></span>` +
        `<span>job: <code>${s.job_id ? s.job_id.slice(-8) : "未启动"}</code></span>` +
      `</div>` +
      clsMiniHtml +
      statusNoteHtml +
      expBtnsHtml;

    if (s.slot_role === "separate_qwen3_enhancement") {
      addonGrid.appendChild(col);
    } else {
      baseGrid.appendChild(col);
    }

    if (isCompleted && s.job_id) {
      const fBtn = document.createElement("button");
      fBtn.type = "button";
      fBtn.className = (LIVE_JOB_ID === s.job_id) ? "primary" : "";
      fBtn.style.padding = "4px 9px";
      fBtn.style.fontSize = "11.5px";
      fBtn.textContent = `检视 ${s.model_display_name} [${s.mode}] (${s.kept_predictions_count}框)`;
      fBtn.onclick = () => focusComparisonSlot(s.job_id);
      focusBar.appendChild(fBtn);
    }
  }

  document.querySelectorAll("[data-focus-job]").forEach(btn => {
    btn.onclick = () => focusComparisonSlot(btn.getAttribute("data-focus-job"));
  });
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

  const out = r.outcome || {};
  document.getElementById("liveResultHeading").textContent =
    `焦点模型详情：${r.model_display_name} [${r.mode_display_name}] (${r.job_id}) — ${out.outcome_label || "已完成"}`;

  const cards = [
    ["模型 / 模式", `${r.model_display_name} / ${r.mode}`],
    ["检测结果状态", out.outcome_label || (r.kept_predictions_count > 0 ? `检出 ${r.kept_predictions_count} 框` : "无有效框")],
    ["模型预测框总数", `${r.kept_predictions_count} (非真实器件数)`],
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
      tr.innerHTML = `<td>${CLASS_CN[cls]} (${cls})</td><td>${gtCnt}</td><td>${c.pred_count || 0}</td><td>${c.tp || 0}/${c.fp || 0}/${c.fn || 0}</td><td>${pct(c.precision)}</td><td>${cr}</td><td>${cf}</td>`;
      gcb.appendChild(tr);
    }
  } else {
    cards.push(["真值与精度评价", "模型预测 / 未做精度评价"]);
    document.getElementById("gtPerClassWrap").style.display = "none";
  }
  document.getElementById("liveKpis").innerHTML = cards
    .map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");

  document.getElementById("liveNote").textContent =
    `${out.outcome_note || ""} | ${r.score_semantics_note} | 正常 EOS=${r.termination_v2.normal_eos}/${r.termination_v2.total_calls}，达上限未闭合=${r.termination_v2.length_limited_without_eos}，显式空数组=${r.termination_v2.explicit_empty}。`;

  const cb = document.getElementById("liveClassBody");
  cb.innerHTML = "";
  for (const cls of CLASSES) {
    const tr = document.createElement("tr");
    const preCnt = r.pre_verifier_class_counts ? r.pre_verifier_class_counts[cls] : "-";
    tr.innerHTML = `<td><span class="cls-swatch" style="background:${CLASS_COLORS[cls]};margin-right:6px;"></span>${CLASS_CN[cls]} (<code>${cls}</code>)</td><td><strong>${r.class_counts[cls] || 0}</strong></td><td>${preCnt}</td>`;
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
  const clsSet = activeClassSet();
  const view = document.getElementById("liveOverlayView").value;
  let list = [];
  if (view === "pre_verifier" && LIVE_RESULT.pre_verifier_candidates) {
    list = LIVE_RESULT.pre_verifier_candidates.map(c => ({ ...c, _tag: c.kept ? "保留" : `剔除(${c.verifier_label || "异常"})` }));
  } else {
    const base = LIVE_RESULT.post_hoc_eval ? LIVE_RESULT.post_hoc_eval.predictions_matched : LIVE_RESULT.predictions;
    list = base.map(p => ({ ...p, _tag: p.match_status || "预测框" }));
    if (view === "with_rejected" && LIVE_RESULT.pre_verifier_candidates) {
      for (const c of LIVE_RESULT.pre_verifier_candidates) {
        if (!c.kept) list.push({ ...c, _rejected: true, _tag: `过滤:${c.verifier_label || "异常"}` });
      }
    }
  }
  return list.filter(it => clsSet.has(it.label));
}

function renderLiveItemsTable() {
  const tbody = document.getElementById("liveItemsBody");
  tbody.innerHTML = "";
  if (!LIVE_RESULT) return;
  const rox = LIVE_RESULT.roi.offset_x, roy = LIVE_RESULT.roi.offset_y;
  const items = getLiveRenderList();
  if (!items.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" style="color:var(--muted);text-align:center;padding:12px;">当前模型或类别筛选下无有效预测框（“无有效框”不代表电路板上无元器件）</td>`;
    tbody.appendChild(tr);
    return;
  }
  for (const item of items) {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    const gx1 = Math.round(item.bbox_xyxy[0]), gy1 = Math.round(item.bbox_xyxy[1]);
    const gx2 = Math.round(item.bbox_xyxy[2]), gy2 = Math.round(item.bbox_xyxy[3]);
    const rx1 = Math.round(item.bbox_xyxy[0] - rox), ry1 = Math.round(item.bbox_xyxy[1] - roy);
    const rx2 = Math.round(item.bbox_xyxy[2] - rox), ry2 = Math.round(item.bbox_xyxy[3] - roy);
    const bcls = item._rejected || item._tag === "FP" || item._tag.startsWith("剔除") ? "b-rej" : "b-keep";
    const cn = CLASS_CN[item.label] || item.label;
    tr.innerHTML =
      `<td><span class="badge ${bcls}">${item._tag}</span></td>` +
      `<td><span class="cls-swatch" style="background:${CLASS_COLORS[item.label] || '#38bdf8'};margin-right:5px;"></span>${cn} (<code>${item.label}</code>)</td>` +
      `<td><code>[${rx1}, ${ry1}, ${rx2}, ${ry2}]</code></td>` +
      `<td><code>[${gx1}, ${gy1}, ${gx2}, ${gy2}]</code></td>`;
    tr.onclick = () => {
      const pad = 280;
      LIVE_CROP = [
        Math.max(0, rx1 - pad),
        Math.max(0, ry1 - pad),
        Math.min(LIVE_RESULT.roi.width, rx2 + pad),
        Math.min(LIVE_RESULT.roi.height, ry2 + pad),
      ];
      renderLiveCanvas();
      renderComparisonPanel();
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
  const coordSpace = document.getElementById("liveCoordSpace").value;

  const roiW = LIVE_UPLOAD.roi_width, roiH = LIVE_UPLOAD.roi_height;
  const rox = LIVE_UPLOAD.roi_offset_x, roy = LIVE_UPLOAD.roi_offset_y;
  const origW = LIVE_UPLOAD.orig_wh ? LIVE_UPLOAD.orig_wh[0] : (roiW + rox);
  const origH = LIVE_UPLOAD.orig_wh ? LIVE_UPLOAD.orig_wh[1] : (roiH + roy);

  const roiObj = { offset_x: rox, offset_y: roy, width: roiW, height: roiH };
  const showView = document.getElementById("liveOverlayView").value;
  const preds = LIVE_RESULT
    ? (LIVE_RESULT.post_hoc_eval ? LIVE_RESULT.post_hoc_eval.predictions_matched : LIVE_RESULT.predictions)
    : [];
  const preCands = LIVE_RESULT ? LIVE_RESULT.pre_verifier_candidates : null;
  const gts = (LIVE_RESULT && LIVE_RESULT.post_hoc_eval) ? LIVE_RESULT.post_hoc_eval.gts : null;

  const ov = buildOverlaySvgMarkup(preds, preCands, gts, roiObj, [origW, origH], coordSpace, LIVE_CROP, showView);

  document.getElementById("liveCanvasTitle").textContent = LIVE_RESULT
    ? `主检视画布：${LIVE_RESULT.safe_filename} — ${LIVE_RESULT.model_display_name} [${LIVE_RESULT.mode}]（可见框 ${ov.visibleCount}/${LIVE_RESULT.kept_predictions_count}）`
    : `预检图像预览：${LIVE_UPLOAD.safe_filename}（原图 ${origW}×${origH} | ROI ${roiW}×${roiH}，${LIVE_UPLOAD.tile_count} 切片）`;

  if (LIVE_CROP) {
    const gx1 = LIVE_CROP[0] + rox, gy1 = LIVE_CROP[1] + roy, gx2 = LIVE_CROP[2] + rox, gy2 = LIVE_CROP[3] + roy;
    document.getElementById("liveCropStatus").textContent =
      `当前局部放大视窗（主画布与三模型对比画布已同步对齐）：ROI 坐标 [${LIVE_CROP.join(", ")}] px | 原图全局坐标 [${gx1}, ${gy1}, ${gx2}, ${gy2}] px`;
  } else {
    document.getElementById("liveCropStatus").textContent = coordSpace === "orig"
      ? `完整原图视野：${origW} × ${origH} px（虚线框标示 ROI 区域 [${rox}, ${roy}, ${rox + roiW}, ${roy + roiH}]）`
      : `完整 ROI 视野：${roiW} × ${roiH} px（相对原图偏移 offset=[${rox}, ${roy}]）`;
  }

  let cropQ = "";
  if (LIVE_CROP) {
    if (coordSpace === "orig") {
      cropQ = `&crop=${LIVE_CROP[0] + rox},${LIVE_CROP[1] + roy},${LIVE_CROP[2] + rox},${LIVE_CROP[3] + roy}`;
    } else {
      cropQ = `&crop=${LIVE_CROP[0]},${LIVE_CROP[1]},${LIVE_CROP[2]},${LIVE_CROP[3]}`;
    }
  }
  img.src = `/api/upload_preview/${encodeURIComponent(LIVE_UPLOAD.upload_id)}?source=${encodeURIComponent(coordSpace)}&max_width=${LIVE_CROP ? 1200 : 1800}${cropQ}`;
  const dispW = zoomVal === "fit" ? Math.max(460, wrap.clientWidth - 4) : Math.min(2800, Math.round(ov.vw * Number(zoomVal)));
  const dispH = Math.round(dispW * (ov.vh / ov.vw));
  img.style.width = dispW + "px";
  img.style.height = dispH + "px";
  stage.style.width = dispW + "px";
  stage.style.height = dispH + "px";
  svg.setAttribute("viewBox", ov.viewBox);
  svg.innerHTML = ov.svgHtml;
}

document.getElementById("liveOverlayView").onchange = () => { renderLiveItemsTable(); renderLiveCanvas(); };
document.getElementById("liveZoom").onchange = renderLiveCanvas;
document.getElementById("btnLiveResetCrop").onclick = () => {
  LIVE_CROP = null;
  renderLiveCanvas();
  renderComparisonPanel();
};

// Downloads — strictly bound to the currently rendered task & active upload
function dlExport(kind) {
  if (!LIVE_JOB_ID || !LIVE_RESULT || LIVE_RESULT.job_id !== LIVE_JOB_ID) return;
  if (LIVE_UPLOAD && LIVE_RESULT.upload_id !== LIVE_UPLOAD.upload_id) return;
  const view = document.getElementById("liveOverlayView").value;
  window.location.href = `/api/jobs/${encodeURIComponent(LIVE_RESULT.job_id)}/export/${kind}?view=${encodeURIComponent(view)}`;
}
document.getElementById("btnExpJson").onclick = () => dlExport("json");
document.getElementById("btnExpPng").onclick = () => dlExport("overlay_png");
document.getElementById("btnExpMd").onclick = () => dlExport("summary_md");

document.getElementById("btnExpCmpMd").onclick = () => {
  if (!LIVE_COMPARISON_ID || !LIVE_COMPARISON || LIVE_COMPARISON.comparison_id !== LIVE_COMPARISON_ID) return;
  if (LIVE_UPLOAD && LIVE_COMPARISON.upload_id !== LIVE_UPLOAD.upload_id) return;
  window.location.href = `/api/comparisons/${encodeURIComponent(LIVE_COMPARISON.comparison_id)}/export/markdown`;
};
document.getElementById("btnExpCmpJson").onclick = () => {
  if (!LIVE_COMPARISON_ID || !LIVE_COMPARISON || LIVE_COMPARISON.comparison_id !== LIVE_COMPARISON_ID) return;
  if (LIVE_UPLOAD && LIVE_COMPARISON.upload_id !== LIVE_UPLOAD.upload_id) return;
  window.location.href = `/api/comparisons/${encodeURIComponent(LIVE_COMPARISON.comparison_id)}/export/json`;
};

// Optional Post-Hoc GT evaluation
document.getElementById("btnEvalGt").onclick = async () => {
  if (!LIVE_JOB_ID || !LIVE_RESULT || LIVE_RESULT.job_id !== LIVE_JOB_ID) return;
  const fIn = document.getElementById("fileGtCoco");
  const msgEl = document.getElementById("gtEvalMsg");
  msgEl.style.display = "none";
  if (!fIn.files || !fIn.files.length) {
    msgEl.textContent = "请先选择事后对比用的 COCO JSON 文件。";
    msgEl.style.display = "block";
    return;
  }
  const session = LIVE_ACTIVE_SESSION;
  const targetJobId = LIVE_JOB_ID;
  const targetCmpId = LIVE_COMPARISON_ID;
  const targetUploadId = LIVE_UPLOAD ? LIVE_UPLOAD.upload_id : null;
  const targetUploadSha = LIVE_UPLOAD ? LIVE_UPLOAD.source_image_sha256 : null;
  const focusSeqAtStart = LIVE_FOCUS_SEQ;
  try {
    const cocoObj = JSON.parse(await fIn.files[0].text());
    if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha }) || LIVE_JOB_ID !== targetJobId || LIVE_FOCUS_SEQ !== focusSeqAtStart) {
      return;
    }
    const imgIdRaw = document.getElementById("inpGtImageId").value.trim();
    const confirmManual = document.getElementById("chkGtManualConfirm").checked;
    const manualReason = document.getElementById("inpGtManualReason").value.trim();
    const body = { coco_gt: cocoObj, confirm_manual_mapping: confirmManual };
    if (imgIdRaw) body.target_image_id = Number(imgIdRaw);
    if (manualReason) body.manual_mapping_reason = manualReason;
    const evalRes = await apiJson(`/api/jobs/${encodeURIComponent(targetJobId)}/evaluate_gt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha }) || LIVE_JOB_ID !== targetJobId || LIVE_FOCUS_SEQ !== focusSeqAtStart) {
      return;
    }
    LIVE_RESULT = evalRes;
    renderLiveResult();
    if (targetCmpId && LIVE_COMPARISON_ID === targetCmpId) {
      const updatedCmp = await apiJson(`/api/comparisons/${encodeURIComponent(targetCmpId)}`);
      if (!isLiveSessionCurrent(session, { uploadId: targetUploadId, uploadSha256: targetUploadSha, comparisonId: targetCmpId })) {
        return;
      }
      if (updatedCmp.comparison_id === targetCmpId) {
        LIVE_COMPARISON = updatedCmp;
        renderComparisonPanel();
      }
    }
  } catch (err) {
    if (!isLiveSessionCurrent(session) || LIVE_JOB_ID !== targetJobId) return;
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
  const reqSeq = ++RES_REQUEST_SEQ;
  const dsKey = document.getElementById("resDataset").value;
  const bid = document.getElementById("resBoard").value;
  const mk = document.getElementById("resModel").value;
  const mode = document.getElementById("resMode").value;
  try {
    const boardData = await apiJson(`/api/research/board?dataset=${encodeURIComponent(dsKey)}&board=${encodeURIComponent(bid)}&model=${encodeURIComponent(mk)}&mode=${encodeURIComponent(mode)}`);
    if (reqSeq !== RES_REQUEST_SEQ) return;
    RES_CURRENT = boardData;
    renderResView();
  } catch (err) {
    if (reqSeq !== RES_REQUEST_SEQ) return;
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
    tr.innerHTML = `<td><span class="badge ${bcls}">${it.k}</span></td><td>${CLASS_CN[it.l] || it.l} (${it.l})</td><td>${it.iou ? "IoU=" + it.iou + " " : ""}[${rx1},${ry1},${rx2},${ry2}]</td>`;
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
  const dispW = zoomVal === "fit" ? Math.max(460, wrap.clientWidth - 4) : Math.min(2800, Math.round(vw * Number(zoomVal)));
  const dispH = Math.round(dispW * (vh / vw));
  const errEl = document.getElementById("resErr");
  if (r.roi && r.roi.image_available === false) {
    img.removeAttribute("src");
    img.style.display = "none";
    stage.style.background = "#0b1220";
    errEl.textContent = "提示：当前轻量离线交付包未附带外置全分辨率 FICS 原板背景图；矢量检测框、指标统计、TP/FP/FN 明细、原始输出与 SHA-256 溯源均可离线完整检视，挂载外置原图后可恢复底图显示。";
    errEl.style.display = "block";
  } else {
    img.style.display = "block";
    stage.style.background = "";
    img.src = `/api/research/image?dataset=${encodeURIComponent(r.dataset_key)}&board=${encodeURIComponent(r.board_id)}&max_width=${RES_CROP ? 1200 : 1800}${cropQ}`;
  }
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
    sel.innerHTML = '<option value="">（载入近期单模型任务记录）</option>';
    for (const j of d.jobs || []) {
      const opt = document.createElement("option");
      opt.value = j.job_id;
      const outLbl = (j.outcome && j.outcome.outcome_label) || j.status;
      opt.textContent = `${j.job_id} | ${j.safe_filename || j.board_id} | ${j.model_key}/${j.mode} [${outLbl}]`;
      sel.appendChild(opt);
    }
    if (cur) sel.value = cur;
  } catch (_) {}
}

async function refreshRecentComparisons() {
  try {
    const d = await apiJson("/api/comparisons");
    const sel = document.getElementById("selRecentCmp");
    const cur = sel.value;
    sel.innerHTML = '<option value="">（载入近期同图多模型对比记录）</option>';
    for (const c of d.comparisons || []) {
      const opt = document.createElement("option");
      opt.value = c.comparison_id;
      opt.textContent = `${c.comparison_id} | ${c.safe_filename} [${c.baseline_mode}] (${(c.model_keys || []).join("+")}) [${c.status}]`;
      sel.appendChild(opt);
    }
    if (cur) sel.value = cur;
  } catch (_) {}
}

document.getElementById("btnLoadRecentCmp").onclick = async () => {
  const cid = document.getElementById("selRecentCmp").value;
  if (!cid) return;
  const errBox = document.getElementById("jobErrorBox");
  errBox.style.display = "none";
  const loadSession = beginLiveSession({ kind: "comparison", comparisonId: cid, isRunning: false });
  const fetchOpts = loadSession.abortController ? { signal: loadSession.abortController.signal } : undefined;
  try {
    const cmp = await apiJson(`/api/comparisons/${encodeURIComponent(cid)}`, fetchOpts);
    if (!isLiveSessionCurrent(loadSession)) return;
    const up = await apiJson(`/api/uploads/${encodeURIComponent(cmp.upload_id)}`, fetchOpts);
    if (!isLiveSessionCurrent(loadSession)) return;
    clearLiveResultsForNewUpload(false);
    LIVE_UPLOAD = up;
    loadSession.uploadId = up.upload_id;
    loadSession.uploadSha256 = up.source_image_sha256;
    loadSession.comparisonId = cmp.comparison_id;
    const cmpRunning = !isComparisonTerminalStatus(cmp.status);
    loadSession.isRunning = cmpRunning;
    loadSession.terminalLocked = !cmpRunning;
    LIVE_COMPARISON_ID = cmp.comparison_id;
    LIVE_COMPARISON = cmp;
    renderUploadMetaBox(LIVE_UPLOAD);
    syncModeOptions();
    syncActionButtonsForSession(loadSession);
    document.getElementById("jobProgressCard").style.display = "block";
    updateComparisonProgressUi(cmp);
    renderComparisonPanel();
    const completedSlots = (cmp.slots || []).filter(s => s.queue_status === "completed" && s.has_result && s.job_id);
    if (completedSlots.length > 0) {
      const pick = completedSlots.find(s => s.kept_predictions_count > 0) || completedSlots[0];
      await focusComparisonSlot(pick.job_id, loadSession);
      if (!isLiveSessionCurrent(loadSession, { comparisonId: cmp.comparison_id })) return;
    } else {
      renderLiveCanvas();
    }
    if (loadSession.isRunning) {
      startSessionPolling(loadSession, pollComparison, 800);
    }
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (!isLiveSessionCurrent(loadSession)) return;
    errBox.textContent = "载入同图对比记录失败：" + err.message;
    errBox.style.display = "block";
  }
};

document.getElementById("btnLoadRecentJob").onclick = async () => {
  const jid = document.getElementById("selRecentJob").value;
  if (!jid) return;
  const errBox = document.getElementById("jobErrorBox");
  errBox.style.display = "none";
  const loadSession = beginLiveSession({ kind: "job", jobId: jid, isRunning: false });
  const fetchOpts = loadSession.abortController ? { signal: loadSession.abortController.signal } : undefined;
  try {
    const st = await apiJson(`/api/jobs/${encodeURIComponent(jid)}`, fetchOpts);
    if (!isLiveSessionCurrent(loadSession)) return;
    const up = await apiJson(`/api/uploads/${encodeURIComponent(st.upload_id)}`, fetchOpts);
    if (!isLiveSessionCurrent(loadSession)) return;
    clearLiveResultsForNewUpload(false);
    LIVE_UPLOAD = up;
    loadSession.uploadId = up.upload_id;
    loadSession.uploadSha256 = up.source_image_sha256;
    loadSession.jobId = st.job_id;
    const jobRunning = !isJobTerminalStatus(st.status);
    loadSession.isRunning = jobRunning;
    loadSession.terminalLocked = !jobRunning;
    LIVE_JOB_ID = st.job_id;
    renderUploadMetaBox(LIVE_UPLOAD);
    syncModeOptions();
    syncActionButtonsForSession(loadSession);
    document.getElementById("jobProgressCard").style.display = "block";
    updateProgressUi(st);
    if (st.outcome) {
      setModelLeftBadge(st.model_key, st.outcome.outcome_label, outcomeBadgeClass(st.outcome.outcome_code));
    }
    if (st.has_result) {
      const res = await apiJson(`/api/jobs/${encodeURIComponent(jid)}/result`, fetchOpts);
      if (!isLiveSessionCurrent(loadSession, { jobId: st.job_id })) return;
      LIVE_RESULT = res;
      document.getElementById("btnEvalGt").disabled = false;
      renderLiveResult();
    } else if (jobRunning) {
      renderLiveCanvas();
      startSessionPolling(loadSession, pollJob, 800);
    } else {
      renderLiveCanvas();
      errBox.textContent = st.error_message || ("任务状态：" + st.status);
      errBox.style.display = "block";
    }
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (!isLiveSessionCurrent(loadSession)) return;
    errBox.textContent = "载入任务失败：" + err.message;
    errBox.style.display = "block";
  }
};

syncModeOptions();
refreshRecentJobs();
refreshRecentComparisons();
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
            elif path == "/api/jobs":
                body = json.loads(self._read_body().decode("utf-8"))
                out = self.service.submit_job(
                    upload_id=str(body.get("upload_id") or ""),
                    model_key=str(body.get("model_key") or ""),
                    mode=str(body.get("mode") or ""),
                )
                self._send_json(HTTPStatus.ACCEPTED, out)
            elif path == "/api/comparisons":
                body = json.loads(self._read_body().decode("utf-8"))
                out = self.service.submit_comparison(
                    upload_id=str(body.get("upload_id") or ""),
                    model_keys=list(body.get("model_keys") or []),
                    mode=str(body.get("mode") or "W"),
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
