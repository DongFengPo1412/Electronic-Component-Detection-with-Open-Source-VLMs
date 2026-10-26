"""
Interactive Local Web Demo Application (Stage 18).
Electronic Component Detection with Open-Source VLMs (Qwen2-VL-2B + LoRA + Tiling).
Built with native Python http.server (no extra dependencies required).
Listens on localhost (127.0.0.1:7860).
"""

import os
import sys
import json
import time
import base64
import io
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pcb_vlm.models.qwen_adapter import QwenAdapter
from pcb_vlm.geometry.tiling import generate_tiles, map_tile_box_to_global, nms_merge_predictions

# Global model holder for lazy loading
LOADED_MODEL = {
    "adapter": None,
    "has_lora": False,
    "base_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master",
    "lora_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage14_lora_training\best_adapter"
}

CATEGORY_COLORS = {
    "resistor": "#3b82f6",    # Blue
    "capacitor": "#10b981",   # Emerald
    "connector": "#f59e0b",   # Amber
    "ic": "#8b5cf6",          # Purple
    "led": "#ef4444",         # Red
    "button": "#ec4899",      # Pink
    "switch": "#14b8a6",      # Teal
    "diode": "#6366f1",       # Indigo
    "transistor": "#f97316",  # Orange
    "inductor": "#84cc16",    # Lime
    "clock": "#06b6d4"        # Cyan
}

SAMPLE_IMAGES = [
    {
        "id": "STK644_Test",
        "name": "STK644 PCB (Test Set - SMT Dense)",
        "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\STK644\STK644.jpg"
    },
    {
        "id": "DuetWIFI_Test",
        "name": "DuetWIFI PCB (Test Set - Power Connectors)",
        "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\DuetWIFI\DuetWIFI.jpg"
    },
    {
        "id": "EDA-008_Val",
        "name": "EDA-008 Top (Validation Set)",
        "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\EDA-008_Top\EDA-008_Top.jpg"
    }
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>基于开源视觉语言大模型的电子元器件检测系统</title>
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-card: rgba(18, 26, 43, 0.75);
      --bg-card-hover: rgba(28, 38, 62, 0.85);
      --border-color: rgba(66, 99, 150, 0.25);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-cyan: #06b6d4;
      --accent-purple: #818cf8;
      --accent-emerald: #34d399;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      background: radial-gradient(circle at 15% 15%, #111a2e 0%, var(--bg-primary) 70%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1.25rem 2.5rem;
      border-bottom: 1px solid var(--border-color);
      background: rgba(10, 14, 23, 0.8);
      backdrop-filter: blur(12px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .logo-group h1 {
      font-size: 1.35rem;
      font-weight: 700;
      background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
    }
    .logo-group p {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }
    .badge-bar {
      display: flex;
      gap: 0.75rem;
    }
    .badge {
      font-size: 0.75rem;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: var(--accent-blue);
      font-weight: 600;
    }
    .badge.emerald {
      background: rgba(52, 211, 153, 0.1);
      border-color: rgba(52, 211, 153, 0.3);
      color: var(--accent-emerald);
    }
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 1.5rem;
      padding: 1.5rem 2.5rem;
      max-width: 1720px;
      margin: 0 auto;
      width: 100%;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 1.25rem;
      backdrop-filter: blur(8px);
    }
    .panel-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: #e2e8f0;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .control-group {
      margin-bottom: 1.15rem;
    }
    .control-group label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.45rem;
    }
    select, button, input[type="file"] {
      width: 100%;
      padding: 0.65rem 0.85rem;
      border-radius: 8px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      background: rgba(15, 23, 42, 0.8);
      color: var(--text-main);
      font-size: 0.85rem;
      outline: none;
      transition: all 0.2s;
    }
    select:focus, input[type="file"]:focus {
      border-color: var(--accent-blue);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    .switch-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.65rem 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .switch-row span {
      font-size: 0.85rem;
      color: #cbd5e1;
    }
    .btn-primary {
      background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
      color: white;
      font-weight: 600;
      cursor: pointer;
      border: none;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
      transition: transform 0.15s, box-shadow 0.15s;
      margin-top: 0.5rem;
      padding: 0.85rem;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
    }
    .btn-primary:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
    }
    .btn-primary:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-top: 1rem;
    }
    .stat-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.75rem;
      border-radius: 8px;
      text-align: center;
    }
    .stat-val {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--accent-blue);
    }
    .stat-label {
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }
    .viewer-area {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      position: relative;
    }
    .canvas-container {
      flex: 1;
      min-height: 520px;
      background: #050811;
      border: 1px solid var(--border-color);
      border-radius: 12px;
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    #pcbCanvas {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }
    .legend-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      padding: 0.75rem;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.75rem;
      color: #cbd5e1;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      background: rgba(15, 23, 42, 0.5);
    }
    .legend-color {
      width: 10px;
      height: 10px;
      border-radius: 2px;
    }
    .loading-overlay {
      position: absolute;
      inset: 0;
      background: rgba(10, 14, 23, 0.85);
      backdrop-filter: blur(4px);
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 1rem;
      z-index: 40;
    }
    .spinner {
      width: 44px;
      height: 44px;
      border: 3px solid rgba(56, 189, 248, 0.2);
      border-top-color: var(--accent-blue);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <div class="logo-group">
      <h1>电子元器件多模态智能检测系统</h1>
      <p>基于 Qwen2-VL-2B 开源视觉语言模型 + LoRA 领域微调 + 1024 物理切片几何重构</p>
    </div>
    <div class="badge-bar">
      <span class="badge">NVIDIA RTX 4060 8GB</span>
      <span class="badge emerald">Stage 18 演示环境已锁定</span>
    </div>
  </header>

  <main>
    <!-- Left Controls Panel -->
    <div class="card">
      <h2 class="panel-title">⚙️ 检测配置与样本载入</h2>

      <div class="control-group">
        <label>预设 PCB 测试板图库 (独立测试集)</label>
        <select id="sampleSelect" onchange="loadSampleImage()">
          <option value="">-- 选择预置 PCB 图像 --</option>
          <option value="STK644_Test">STK644 板 (未见测试集 - 贴片微器件)</option>
          <option value="DuetWIFI_Test">DuetWIFI 板 (未见测试集 - 功率接插件)</option>
          <option value="EDA-008_Val">EDA-008 Top 板 (验证集)</option>
        </select>
      </div>

      <div class="control-group">
        <label>或从本地上传自定义 PCB 图像</label>
        <input type="file" id="imageUpload" accept="image/*" onchange="handleImageUpload(event)">
      </div>

      <div class="switch-row">
        <span>启用 LoRA 适配器权重 (best_adapter)</span>
        <input type="checkbox" id="loraToggle" checked>
      </div>

      <div class="switch-row">
        <span>启用 1024×1024 重叠分块逆变换 (SAHI)</span>
        <input type="checkbox" id="tilingToggle" checked>
      </div>

      <button id="detectBtn" class="btn-primary" onclick="runDetection()">
        <span>🚀 运行元器件智能检测</span>
      </button>

      <div class="stats-grid">
        <div class="stat-box">
          <div id="statDets" class="stat-val">0</div>
          <div class="stat-label">检出元器件数</div>
        </div>
        <div class="stat-box">
          <div id="statTime" class="stat-val">0.0s</div>
          <div class="stat-label">推理耗时</div>
        </div>
        <div class="stat-box">
          <div id="statVram" class="stat-val">0.0 MB</div>
          <div class="stat-label">显存占用</div>
        </div>
        <div class="stat-box">
          <div id="statResolution" class="stat-val">0×0</div>
          <div class="stat-label">图像物理分辨率</div>
        </div>
      </div>

      <div style="margin-top: 1.25rem;">
        <button onclick="exportJSON()" style="background: rgba(255,255,255,0.05); cursor: pointer; margin-bottom: 0.5rem;">📥 导出检测结果 JSON</button>
      </div>
    </div>

    <!-- Right Viewer Canvas -->
    <div class="viewer-area">
      <div class="canvas-container">
        <canvas id="pcbCanvas"></canvas>
        <div id="loadingOverlay" class="loading-overlay">
          <div class="spinner"></div>
          <p id="loadingText" style="font-size: 0.9rem; color: #cbd5e1;">正在调用 Qwen2-VL 模型进行多切片检测...</p>
        </div>
      </div>

      <!-- Categories Legend -->
      <div id="legendBar" class="legend-bar"></div>
    </div>
  </main>

  <script>
    let currentImage = null;
    let currentDetections = [];
    const CATEGORY_COLORS = {
      "resistor": "#3b82f6",
      "capacitor": "#10b981",
      "connector": "#f59e0b",
      "ic": "#8b5cf6",
      "led": "#ef4444",
      "button": "#ec4899",
      "switch": "#14b8a6",
      "diode": "#6366f1",
      "transistor": "#f97316",
      "inductor": "#84cc16",
      "clock": "#06b6d4"
    };

    function initLegend() {
      const container = document.getElementById('legendBar');
      container.innerHTML = '';
      for (const [cat, color] of Object.entries(CATEGORY_COLORS)) {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<span class="legend-color" style="background:${color}"></span><span>${cat}</span>`;
        container.appendChild(item);
      }
    }
    initLegend();

    function renderImageOnCanvas(img, boxes = []) {
      const canvas = document.getElementById('pcbCanvas');
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      document.getElementById('statResolution').innerText = `${canvas.width}×${canvas.height}`;

      boxes.forEach(b => {
        const [x1, y1, x2, y2] = b.bbox_xyxy;
        const color = CATEGORY_COLORS[b.label] || '#38bdf8';
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(3, Math.round(canvas.width / 800));
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = color;
        const fontSize = Math.max(14, Math.round(canvas.width / 120));
        ctx.font = `bold ${fontSize}px sans-serif`;
        const text = `${b.label}`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(x1, Math.max(0, y1 - fontSize - 6), textWidth + 8, fontSize + 6);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 4, Math.max(fontSize, y1 - 4));
      });
    }

    function loadSampleImage() {
      const val = document.getElementById('sampleSelect').value;
      if (!val) return;
      const overlay = document.getElementById('loadingOverlay');
      overlay.style.display = 'flex';
      document.getElementById('loadingText').innerText = '正在载入预设 PCB 原图...';

      fetch(`/api/sample_image?id=${val}`)
        .then(r => r.json())
        .then(data => {
          overlay.style.display = 'none';
          const img = new Image();
          img.onload = () => {
            currentImage = img;
            currentDetections = [];
            renderImageOnCanvas(img);
            document.getElementById('statDets').innerText = '0';
          };
          img.src = 'data:image/jpeg;base64,' + data.base64;
        })
        .catch(err => {
          overlay.style.display = 'none';
          alert('载入图像失败: ' + err);
        });
    }

    function handleImageUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        const img = new Image();
        img.onload = () => {
          currentImage = img;
          currentDetections = [];
          renderImageOnCanvas(img);
        };
        img.src = ev.target.result;
      };
      reader.readAsDataURL(file);
    }

    function runDetection() {
      if (!currentImage) {
        alert('请先选择或上传一张 PCB 图像！');
        return;
      }
      const overlay = document.getElementById('loadingOverlay');
      overlay.style.display = 'flex';
      document.getElementById('loadingText').innerText = '正在执行大模型元器件检测推理...';
      document.getElementById('detectBtn').disabled = true;

      const canvas = document.createElement('canvas');
      canvas.width = currentImage.naturalWidth || currentImage.width;
      canvas.height = currentImage.naturalHeight || currentImage.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(currentImage, 0, 0);
      const b64 = canvas.toDataURL('image/jpeg', 0.95).split(',')[1];

      const payload = {
        image_base64: b64,
        use_lora: document.getElementById('loraToggle').checked,
        use_tiling: document.getElementById('tilingToggle').checked
      };

      fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(r => r.json())
      .then(res => {
        overlay.style.display = 'none';
        document.getElementById('detectBtn').disabled = false;
        if (res.error) {
          alert('检测异常: ' + res.error);
          return;
        }
        currentDetections = res.objects || [];
        document.getElementById('statDets').innerText = currentDetections.length;
        document.getElementById('statTime').innerText = (res.latency_ms / 1000).toFixed(2) + 's';
        document.getElementById('statVram').innerText = (res.peak_vram_mb || 0).toFixed(1) + ' MB';
        renderImageOnCanvas(currentImage, currentDetections);
      })
      .catch(err => {
        overlay.style.display = 'none';
        document.getElementById('detectBtn').disabled = false;
        alert('请求失败: ' + err);
      });
    }

    function exportJSON() {
      if (!currentDetections.length) {
        alert('当前暂无检测结果！');
        return;
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(currentDetections, null, 2));
      const a = document.createElement('a');
      a.setAttribute('href', dataStr);
      a.setAttribute('download', 'pcb_detections.json');
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  </script>
</body>
</html>
"""


class DemoHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif url.path == "/api/sample_image":
            query = urllib.parse.parse_qs(url.query)
            sample_id = query.get("id", [""])[0]
            matched = next((s for s in SAMPLE_IMAGES if s["id"] == sample_id), None)
            if matched and os.path.exists(matched["path"]):
                with open(matched["path"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"id": sample_id, "base64": b64}).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/detect":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                img_b64 = data.get("image_base64", "")
                use_lora = data.get("use_lora", True)
                use_tiling = data.get("use_tiling", True)

                # Decode image
                raw_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                w, h = pil_img.size

                # Load or switch model
                if LOADED_MODEL["adapter"] is None or LOADED_MODEL["has_lora"] != use_lora:
                    print(f"Loading Qwen2-VL (LoRA={use_lora})...")
                    adapter = QwenAdapter(model_id=LOADED_MODEL["base_path"], device="cuda")
                    if use_lora and os.path.exists(LOADED_MODEL["lora_path"]):
                        from peft import PeftModel
                        adapter.model = PeftModel.from_pretrained(adapter.model, LOADED_MODEL["lora_path"])
                        adapter.model.eval()
                    LOADED_MODEL["adapter"] = adapter
                    LOADED_MODEL["has_lora"] = use_lora

                adapter = LOADED_MODEL["adapter"]
                prompt = adapter.build_prompt()

                t0 = time.time()
                temp_img_path = os.path.join(r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024", "_demo_temp.jpg")
                pil_img.save(temp_img_path)

                if use_tiling:
                    tiles = generate_tiles(w, h, tile_size=1024, overlap_ratio=0.20)
                    patch_predictions = []
                    for t in tiles:
                        bx1, by1, bx2, by2 = t["bbox"]
                        crop_tile = pil_img.crop((bx1, by1, bx2, by2))
                        tile_path = os.path.join(r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024", "_demo_tile.jpg")
                        crop_tile.save(tile_path)
                        tile_res = adapter.predict_image(tile_path, prompt=prompt, max_pixels=768*28*28)
                        if os.path.exists(tile_path):
                            os.remove(tile_path)
                        for obj in tile_res["objects"]:
                            g_box = map_tile_box_to_global(obj["bbox_xyxy"], t["offset_x"], t["offset_y"], w, h)
                            patch_predictions.append({
                                "label": obj["label"],
                                "bbox_xyxy": g_box,
                                "score": obj.get("score")
                            })
                    merged_objects = nms_merge_predictions(patch_predictions, iou_threshold=0.50)
                else:
                    res = adapter.predict_image(temp_img_path, prompt=prompt, max_pixels=768*28*28)
                    merged_objects = res["objects"]

                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

                latency_ms = (time.time() - t0) * 1000
                import torch
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                resp = {
                    "num_detected": len(merged_objects),
                    "objects": merged_objects,
                    "latency_ms": round(latency_ms, 2),
                    "peak_vram_mb": round(vram_mb, 2)
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def main():
    port = 7860
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, DemoHTTPHandler)
    print(f"=== PCB Component Detection Web App ===")
    print(f"Listening on http://127.0.0.1:{port}")
    print(f"Press Ctrl+C to terminate.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
