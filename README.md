# Electronic Component Detection with Open-Source VLMs

面向印刷电路板（PCB）电子元器件定位与分类的开源视觉语言模型（VLM）评测与可复现研究仓库。当前主线聚焦 **FICS-PCB DSLR 6 类开放域多模型对照评测**（`capacitor`、`resistor`、`inductor`、`transistor`、`diode`、`ic`），提供从冻结实验计划、隔离式顺序推理、严格 `v2` 终止/契约解析、`IoU >= 0.50` 类感知评价到只读可视化回放的完整可验证链路。

## 1. 当前主线结论与核心数字（统一 512-Token 实用预算）

在统一任务契约（`xyxy_norm1000` JSON 数组、`max_new_tokens=512`、`do_sample=False`、同卡单步串行与双门限显存检查、不做预检丢弃与失败零填充）下，三款轻量级开源 VLM 在两个严格隔离的 FICS-PCB 样本集上的真实回放结果如下（完整自动生成表见 [docs/fics_results_tables.md](docs/fics_results_tables.md)）：

### 1.1 开发集：`s18_front + s28_front`（201 次正式调用，每模型 67 次，共 64 GT）

| 模型 | 模式 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen2-VL-2B-Instruct` | `W` (整 ROI) | 0 | 11 | 64 | 0.00% | 0.00% | 0.00% |
| `Qwen2-VL-2B-Instruct` | `T` (切片+NMS) | 0 | 355 | 64 | 0.00% | 0.00% | 0.00% |
| `deepseek-vl-1.3b-chat` | `W` (整 ROI) | 0 | 0 | 64 | 0.00% | 0.00% | 0.00% |
| `deepseek-vl-1.3b-chat` | `T` (切片+NMS) | 0 | 6 | 64 | 0.00% | 0.00% | 0.00% |
| `Qwen3-VL-2B-Instruct` | `W` (整 ROI) | 3 | 13 | 61 | 18.75% | 4.69% | 7.50% |
| `Qwen3-VL-2B-Instruct` | `T` (切片+NMS) | **19** | **257** | **45** | **6.88%** | **29.69%** | **11.18%** |

### 1.2 未观察新板样本：`s29 + s27`（306 次正式调用，每模型 102 次，共 41 GT）

| 模型 | 模式 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen2-VL-2B-Instruct` | `W` (整 ROI) | 0 | 10 | 41 | 0.00% | 0.00% | 0.00% |
| `Qwen2-VL-2B-Instruct` | `T` (切片+NMS) | 1 | 550 | 40 | 0.18% | 2.44% | 0.34% |
| `deepseek-vl-1.3b-chat` | `W` (整 ROI) | 0 | 0 | 41 | 0.00% | 0.00% | 0.00% |
| `deepseek-vl-1.3b-chat` | `T` (切片+NMS) | 0 | 0 | 41 | 0.00% | 0.00% | 0.00% |
| `Qwen3-VL-2B-Instruct` | `W` (整 ROI) | 6 | 16 | 35 | **27.27%** | 14.63% | **19.05%** |
| `Qwen3-VL-2B-Instruct` | `T` (切片+NMS) | **12** | 235 | 29 | 4.86% | **29.27%** | 8.33% |

> **关键发现与严格边界声明**
> - **切块（T）并非无条件优于整图（W）**：在 `s29+s27` 上，Qwen3 切块虽然将 Recall 从 `14.63%` 提升至 `29.27%`，但误报由 `16` 激增至 `235`，F1 反从 `19.05%` 降至 `8.33%`。
> - **架构与输入预算差异导致零/低命中**：三模型本身处于不同参数量、视觉编码器分辨率处理（`DeepSeek-VL` 固定 `384x384` 等比填充至 `576` token，而 `Qwen2/Qwen3` 使用受控动态分辨率）及微调/指令对齐阶段，`DeepSeek-VL-1.3B` 在 512-token 下虽然多数调用正常结束 EOS，但几乎全部退化为自然语言描述而非严格 JSON；`Qwen2-VL-2B` 能生成格式框但坐标定位普遍偏移。
> - **数据集口径、硬件与设计系列边界**：本文评测数据来自 **FICS-PCB**（Lu 等人，Cryptology ePrint Archive 2020/366，全集含 31 个样本 `s1..s31`、9,912 张 DSLR 与数字显微图像及 77,347 个标注；**切勿与仓内 `papers/01_FICS-PCB_FPIC_Dataset_2022.pdf` 中实际收录的 FPIC 数据集论文〔Jessurun 等人，arXiv:2202.08414 / ACM JETC 2023〕混淆**）。本项目仅评测其 **DSLR 正面图像子集中的 4 个样本板卡 / 4 张正面图像**（`s18_front` 为深色 IoT/RF MCU 开发板 `PO USI 850104 BM-09-S 573`，非 MK4；`s28_front` 为绿色 `ICLRT-EFI5 E-field Integrator MK5`；`s29` 与 `s27` 为同系列蓝色 `E-field Integrator MK4 RC03 - 2011 ICLRT`；未评测显微镜图像）。`s29+s27` 中 `diode` GT 数为 `0`，故其 Recall 记为 `N/A (GT=0)` 并如实统计 FP。全部 GPU 推理均在 **NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存，`8188.0 MiB`）** 上单模型串行执行，峰值保留显存控制在 `4248–5040 MiB`（`4.1–4.9 GiB`）。不得将 4 块板卡小样本结果外推为“整套 FICS-PCB 数据集广泛泛化”或“统计学显著结论”。

---

## 2. 架构与协议文档及关键模块

- 模型架构与实验协议说明：[docs/model_architectures_and_fics_protocol.md](docs/model_architectures_and_fics_protocol.md)
- 自动汇总结果表（严禁手改数值）：[docs/fics_results_tables.md](docs/fics_results_tables.md)
- FICS 6 类隔离配置：[configs/fics_six_classes.yaml](configs/fics_six_classes.yaml)
- 本地推理运行路径模板：[configs/fics_local_runtime.example.yaml](configs/fics_local_runtime.example.yaml)
- 关键源码模块：
  - `src/pcb_vlm/data/fics_dslr.py` — FICS-PCB 官方元数据、XML/CSV 核对与 6 类隔离 COCO 构造
  - `src/pcb_vlm/evaluation/fics_replay.py` — SHA-256 计划绑定、防覆盖校验、`v2` EOS/预算截断分离解析、逐板 NMS 与 `IoU>=0.50` 评测
  - `src/pcb_vlm/inference/` (`gpu_gate.py`, `fics_worker.py`, `fics_matrix.py`) — 仓库化隔离推理执行器、双门限显存探针、超时/EOF 看门狗与防覆盖保护

---

## 3. 常用命令与复现流程

### 3.1 运行全量 CPU 单测与集成合同测试
```bash
python -m unittest discover -s tests -v
```

### 3.2 可复现 CPU 回放评价（不依赖 GPU / 权重）
```bash
python scripts/replay_fics_eval.py \
  --gt-coco <combined_coco.json> \
  --replay-manifest <replay_identity_manifest.json> \
  --runs-dir <runs_dir> \
  --plan <frozen_plan.json> \
  --output <new_report.json>
```
> 注：默认拒绝覆盖任何已存在文件或输入路径；支持通过 `--path-map OLD=NEW` 与 `--image-root DIR` 在离线证据包中重定位图像并执行 `--verify-disk-hashes`。

### 3.3 集成推理入口验证（默认 CPU `--dry-run` 与显式 `--execute`）
`scripts/run_fics_matrix.py` 由已验证的外部 GPU matrix runner 整合迁移进仓库。基于本地提交 `376011b`，该集成入口已通过全仓 163 项 CPU 合同/回归测试、双份 `--dry-run` 核验，并已在 `RTX 4060 Laptop GPU (8 GB)` 上通过显式 `--execute --verify-input-hashes` 完成了三模型各 `1` 次（`qwen2 -> deepseek -> qwen3` 共 `3` 次调用、墙钟 `93.421 s`、退出码 `0`）的受控 **GPU 工程 Smoke 实测**，实证验证了仓库集成后的模型加载、图像处理、首步生成与 `logits` 有限性检查、原始记录落盘与来源 SHA-256 校验链路（独立审查回执见 `C:\Users\Liu\.codex\artifacts\pcb-fics-integration-smoke-review-20261115\independent_smoke_and_git.json` 及 `审查结论.md`）。

> **集成测试边界声明**：本次 Smoke 仅验证单个冻结切片（`s18_front` `T00`）的端到端工程通路，**未在仓库集成后重跑完整 507 调用主矩阵（第 1 节 201+306 成绩仍来自已验收的正式回放，Smoke 结果不并入主成绩表）**；其中 `Qwen2` 可解析的 `6` 个框与 `Qwen3` 截断前恢复的 `14` 个框均仅代表结构化提取数量而非 `TP` 命中数，`DeepSeek` 在 `145` token 正常 EOS 后返回英文自然语言描述（`0` 个有效框）属于输出格式契约失败而非未执行。

为防止误触 GPU，CLI 与 Python API 现均**默认运行 CPU `--dry-run`**（仅在显式传入 `--execute` / `dry_run=False` 时才执行 GPU 探针与 worker）；且 `--runtime-config` 仅允许覆盖本地路径与运行环境，严禁修改冻结 `repo_id`、`revision`、`attn_implementation`、`visual_budget` 与队列配置。

1. 复制 [configs/fics_local_runtime.example.yaml](configs/fics_local_runtime.example.yaml) 为本地配置（如 `configs/fics_local_runtime.yaml`）并填入本地权重与 `qwen3` 依赖覆盖路径。
2. 先执行 CPU 干跑检查计划、重映射图像与输出目录安全性（可省略 `--dry-run`，显式写明更清晰；与 `--execute` 互斥）：
   ```bash
   python scripts/run_fics_matrix.py \
     --plan <frozen_plan.json> \
     --runtime-config configs/fics_local_runtime.yaml \
     --output-dir <new_clean_output_dir> \
     --dry-run --verify-input-hashes
   ```
3. 未来如需在 GPU 上执行集成入口实跑复验（必须显式传 `--execute`，且强制执行双门限显存探针 `nvidia-smi >= 5500 MiB` 与 `torch.cuda.mem_get_info >= 6144 MiB`）：
   ```bash
   python scripts/run_fics_matrix.py \
     --plan <single_board_smoke_plan.json> \
     --runtime-config configs/fics_local_runtime.yaml \
     --output-dir <new_smoke_output_dir> \
     --models qwen3 \
     --execute
   ```

### 3.4 自动生成汇总表/图表 与 只读真实回放展示页面
```bash
# 1) 根据 results_index.json 校验 SHA-256 并重新生成表与图
python scripts/build_fics_results_bundle.py \
  --index <results_index.json> \
  --out-json <results_summary_tables.json> \
  --out-md docs/fics_results_tables.md \
  --figures-dir <figures_dir>

# 2) 启动只读历史回放展示服务（支持数据集隔离切换、TP/FP/FN 叠加、局部放大与原文检查）
python scripts/fics_showcase.py --index <results_index.json> --port 7861
```

---

## 4. 历史工作归档（WACV 2019 11 类 / QLoRA R6-R7）

以下内容属于早期在 **WACV 2019 11 类 PCB 数据集** 上的基准探索、QLoRA 监督微调（R6 / R7）及对应演示服务，与当前 FICS-PCB 6 类主线完全解耦：
- 类别与任务旧配置：`configs/classes.yaml`、`configs/task.yaml`、`configs/lora*.yaml`
- 旧演示服务入口：`scripts/demo_app.py`（仅面向历史 WACV 模型与旧路径，请勿用于当前 FICS-PCB 6 类结果展示）
- 旧批次评估与训练脚本：`scripts/train_lora.py`、`scripts/infer_batch.py`、`scripts/run_v2_comparison.py`、`scripts/run_v3_lora_comparison.py`、`scripts/train_yolo_v3.py`
