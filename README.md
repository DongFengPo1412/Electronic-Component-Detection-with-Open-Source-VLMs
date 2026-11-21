# Electronic Component Detection with Open-Source VLMs

面向印刷电路板（PCB）电子元器件定位与分类的开源视觉语言模型（VLM）评测、二次候选核验与无真值演示系统仓库。当前最终主线聚焦 **FICS-PCB DSLR 正面图像 6 类开放域多模型对照评测**（`capacitor`、`resistor`、`inductor`、`transistor`、`diode`、`ic`），提供从冻结实验计划、双门限串行推理、`v2` 终止/契约解析、候选框局部裁剪二次核验（`B_other`）、`IoU >= 0.50` 类感知评价，到 **GT-free 新图上传整板检测 + 只读 8 板研究回放**的完整可复现链路。

## 1. 最终主线实验设计、8 板数据划分与核心结论

在统一基础任务契约（`xyxy_norm1000` JSON 数组、`max_new_tokens=512`、`do_sample=False`、同卡单步串行与双门限显存检查、不做预检丢弃与失败零填充）与冻结二阶段局部核验规格（`B_other`：对 `Qwen3` 切片去重候选框做 `10%` 外扩裁剪与单选分类过滤，保留 `other` 以外全部 6 类候选且**严格保留原候选类别、坐标、`score=1.0` 与相对顺序**）下，8 张 FICS-PCB DSLR 正面图像（总计 **375 GT**，均保持原始 64 个设计系列边界）的严格隔离评测结果如下：

### 1.1 历史基准与开发扩展阶段（6 张已观察参考图，共 268 GT；非最终盲测）
- **历史初始参考（`s18_front + s28_front`，64 GT，201 次三模型基础调用）**：Qwen3 切片 `T` 在 64 GT 上取得 `19 TP / 257 FP / 45 FN`（`F1=11.18%`），优于整图 `W`（`7.50%`）与 Qwen2/DeepSeek 基线。
- **前期未观察对照（`s29 + s27`，41 GT，306 次三模型基础调用）**：Qwen3 切片 `T` 虽将 Recall 从 `14.63%` 提升至 `29.27%`，但误报由 `16` 激增至 `235`，导致 F1 反由 `19.05%` 降至 `8.33%`（`diode` GT=0，Recall 记为 `N/A (GT=0)`），暴露高分辨率切片下的严重误报瓶颈。
- **新增开发板方法选型（`s4 + s20`，163 GT，186 次三模型基础调用 + 521 次 Qwen3 二阶段开发核验）**：
  - 基线 `Qwen3 T`：`19 TP / 502 FP / 144 FN`，`P=3.65%`，`R=11.66%`，`F1=5.56%`；
  - 方案 `A`（切片一致性跨切片投票过滤）：`14 TP / 188 FP / 149 FN`，`P=6.93%`，`R=8.59%`，`F1=7.67%`；
  - **预注册胜出并冻结方案 `B_other`**：`15 TP / 156 FP / 148 FN`，`P=8.77%`，`R=9.20%`，`F1=8.98%`（按预注册未舍入 micro-F1 胜出并冻结 10 项核心代码/配置 SHA-256）；
  - 方案 `B_exact`（严格同类确认，阴性对照）：`9 TP / 51 FP / 154 FN`，`P=15.00%`，`R=5.52%`，`F1=8.07%`（召回损失过大）。

### 1.2 最终独立保留盲测集（`s19 + s16`，107 GT；单次盲评，未作任何事后调参）
三模型各自执行 `W=2, T=140`（batch 合计 `142` 次，三模型基础矩阵共 `426` 次调用）；冻结主方法 `Qwen3_T_plus_B_other` 实际执行 **`140` 次切片推理 + `508` 次去重候选裁剪核验**（剔除 `358` 个背景/非元器件 `other` 候选，保留 **`150`** 个最终预测框）：

| 模型 | 模式 / 方法 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen2-VL-2B-Instruct` | `W` (整 ROI) | 0 | 11 | 107 | 0.00% | 0.00% | 0.00% |
| `Qwen2-VL-2B-Instruct` | `T` (切片+NMS) | 0 | 895 | 107 | 0.00% | 0.00% | 0.00% |
| `deepseek-vl-1.3b-chat` | `W` (整 ROI) | 0 | 0 | 107 | 0.00% | 0.00% | 0.00% |
| `deepseek-vl-1.3b-chat` | `T` (切片+NMS) | 0 | 0 | 107 | 0.00% | 0.00% | 0.00% |
| `Qwen3-VL-2B-Instruct` | `W` (整 ROI) | 4 | 25 | 103 | **13.79%** | 3.74% | 5.88% |
| `Qwen3-VL-2B-Instruct` | `T` (切片+NMS) | **12** | 496 | **95** | 2.36% | **11.21%** | 3.90% |
| `Qwen3-VL-2B-Instruct` | **`T + B_other`（冻结主方法）** | 10 | **140** | 97 | 6.67% | 9.35% | **7.78%** (`7.7821%`) |

> **关键研究结论与严格边界声明**
> - **误报抑制收益与召回代价并存**：在未见过任何参数调整的 `s19+s16` 上，`B_other` 将 `Qwen3 T` 的误报由 `496` 削减至 `140`（降幅 `71.77%`），Precision 由 `2.36%` 升至 `6.67%`，F1 由 `3.90%` 升至 `7.7821%`（同时超过 `W` 的 `5.88%`），但代价是真阳性由 `12` 降至 `10`（损失 `2` 个 TP）。开源轻量 VLM 在高密度 PCB 小目标零样本检测上仍远未达到工业可用精度。
> - **调用计量透明**：基础研究矩阵 4 批合计 `201 + 306 + 186 + 426 = 1119` 次基础调用（另含开发核验 `521` 次、最终保留核验 `508` 次、早期 `24` 项诊断、`3` 次集成 Smoke 与功能演示实时验证），全部实验均未进行任何权重微调（训练次数 `= 0`）。
> - **数据集与硬件边界**：评测数据来自 **FICS-PCB**（Lu 等人，Cryptology ePrint Archive 2020/366，全集含 31 样本、9,912 张 DSLR/显微图，请勿与仓内 `papers/01_FICS-PCB_FPIC_Dataset_2022.pdf` 收录的 FPIC 数据集论文混淆）。本文仅覆盖 **DSLR 正面子集 8 张图像（375 GT）**；所有推理在 **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, 8188 MiB)** 单卡串行完成，基础模型 `Qwen2/DeepSeek` 运行于 `transformers 4.46.3`，`Qwen3` 使用 `transformers 4.57.6` 隔离覆盖路径（`torch 2.5.1+cu121`）。不得将 2 块保留板差异外推为任意 PCB 设计家族通用泛化结论。

---

## 2. 架构、核心模块与演示闭环

- 模型架构与实验协议说明：[docs/model_architectures_and_fics_protocol.md](docs/model_architectures_and_fics_protocol.md)
- 自动汇总结果表（历史 4 板参考）：[docs/fics_results_tables.md](docs/fics_results_tables.md)
- FICS 6 类隔离配置：[configs/fics_six_classes.yaml](configs/fics_six_classes.yaml)
- 模型本地权重与覆盖路径模板：[configs/fics_local_runtime.example.yaml](configs/fics_local_runtime.example.yaml)（对应 `--runtime-config`，含 `models` 与 `gpu_gate`）
- 最终演示服务端启动配置模板：[configs/fics_final_demo.example.json](configs/fics_final_demo.example.json)（对应 `--config`，仅负责 `workspace_root`、`runtime_config`、`results_index`、`host`、`port`）
- 关键源码模块：
  - `src/pcb_vlm/data/fics_dslr.py` — FICS-PCB 官方元数据解析与 6 类隔离 COCO 构造
  - `src/pcb_vlm/geometry/tiling.py` — GT-free 白边 ROI 检测（像素 `<248 + 32px` 边距）与 `1024px / 20%` 切片生成
  - `src/pcb_vlm/evaluation/fics_replay.py` — SHA-256 计划绑定、`v2` 终止分离解析、`IoU >= 0.50` 类感知评价
  - `src/pcb_vlm/evaluation/tile_consistency.py` & `scripts/eval_tile_consistency.py` — 开发集候选对照方法 `A`（跨切片一致性过滤）
  - `src/pcb_vlm/evaluation/candidate_verifier.py` & `src/pcb_vlm/inference/candidate_verifier_runner.py` & `scripts/run_candidate_verifier.py` — 冻结方法 `B_other`（候选框 10% 外扩裁剪与二阶段分类核验）
  - `src/pcb_vlm/evaluation/gt_free_pipeline.py` — 无 GT 上传图重映射、切片去重与 `B_other` 过滤导出链路
  - `src/pcb_vlm/evaluation/final_demo_service.py` & `scripts/fics_final_demo.py` — 完整端到端演示后端与双页签 Web UI（无 GT 新图上传、单模型与三模型同图串行对比队列 `/api/comparisons`、6 类中英文筛选与颜色图例、原图/ROI 坐标对齐切换、轻量同图对比报告导出、前端会话代号与 `AbortController` 防跨任务迟到响应串台/停轮询、同会话单飞与终态锁防重叠轮询回退、只读回放示例前后端双重重推拦截、`--config relative_to` 便携路径锚点、后端取消与派发原子锁、可安全取消子进程树、可选事后 GT 身份核验、只读 8 板研究回放与离线轻量包只读回放降级提示）

---

## 3. 常用命令与复现流程

### 3.1 运行 CPU 合同与回归测试
```bash
# 运行核心单元测试
python -m unittest discover -s tests -v

# 运行演示服务、无 GT 管道与方法对比 pytest 套件
python -m pytest tests/test_final_demo.py tests/test_gt_free_pipeline.py tests/test_candidate_verifier.py tests/test_tile_consistency.py -v
```
> 若同时设置环境变量 `FICS_HELDOUT_ARTIFACT_DIR` 与 `FICS_RESULTS_INDEX` 指向真实 Held-Out/8 板索引产物，可追加触发对应真实产物 CPU 重建完整性回归测试。

### 3.2 启动最终端到端演示与 8 板回放系统（GT-free）
1. **模型路径配置**（仅 `--runtime-config` YAML）：复制 `configs/fics_local_runtime.example.yaml` 至本地仓库外路径并填写 `qwen2` / `deepseek` / `qwen3` 权重目录及 `qwen3` `overlay_dir`。
2. **可选服务启动 JSON**（`--config` JSON，命令行参数优先级高于 JSON 字段）：复制 `configs/fics_final_demo.example.json` 并填入本机 `workspace_root`、`runtime_config`、`results_index`。
3. **启动服务端**（若仅做 8 板只读回放与功能测试，可省略 `--runtime-config`）：
```bash
# 方式 A：使用 --config 启动，必要时用 CLI 参数覆盖
python scripts/fics_final_demo.py --config /path/to/local_demo_startup.json --port 7862

# 方式 B：直接通过命令行参数启动
python scripts/fics_final_demo.py \
  --workspace-root /path/to/demo_workspace \
  --runtime-config /path/to/local_runtime.yaml \
  --results-index /path/to/results_index.json \
  --host 127.0.0.1 --port 7862
```
- 新图上传推理全过程**严格不读取 Ground Truth**（结果页默认标记“模型预测 / 未做精度评价”，不把预测框数等同于真实器件数，不展示 `score=1.0` 为 100% 置信度）；支持单模型运行与 **三模型（`Qwen2-VL-2B-Instruct`、`deepseek-vl-1.3b-chat（v1）`、`Qwen3-VL-2B-Instruct`）同图串行队列对比**，界面显式区分“完成（检出预测框）”、“无有效框（不等于板上无元器件）”与“运行失败/已取消（不生成伪造空成绩）”。
- 事后上传可选 COCO GT 时，系统校验类别、像素尺寸及声明 SHA-256（**已知哈希冲突直接拒绝**）；若无 SHA 且文件名不一致（即使单图 COCO 或指定了 `target_image_id`），必须由用户显式勾选人工确认并如实标记 `image_identity_verified=false`。当某类 GT 支持数为 `0` 时，Recall 与 F1 规范输出 `null` / `N/A (GT=0)` 并保留真实 `FP` 计数。

### 3.3 集成推理入口与离线 CPU 回放
- **集成推理 CLI（默认安全 `--dry-run`）**：`python scripts/run_fics_matrix.py --plan <frozen_plan.json> --runtime-config <local_runtime.yaml> --output-dir <new_dir> [--execute]`
- **二阶段核验 CLI（默认安全 `--dry-run`）**：`python scripts/run_candidate_verifier.py --manifest <verifier_manifest.json> --output-dir <new_dir> [--execute]`
- **可复现 CPU 回放评价**：`python scripts/replay_fics_eval.py --gt-coco <combined_coco.json> --replay-manifest <replay_identity_manifest.json> --runs-dir <runs_dir> --plan <frozen_plan.json> --output <new_report.json>`

### 3.4 外部运行依赖说明
仓库与小型 CPU 证据包均不包含数 GB 级本地预训练权重、`transformers 4.57.6` 覆盖包及 FICS-PCB 原图。真实 GPU 推理需预先在本地准备：`Qwen/Qwen2-VL-2B-Instruct`（本地 `snapshots/master`，`revision=null`）、`deepseek-ai/deepseek-vl-1.3b-chat`（本地 `snapshots/master`，`revision=null`）、`Qwen/Qwen3-VL-2B-Instruct`（commit `89644892e4d85e24eaac8bacfd4f463576704203`），以及 CUDA 12.1 环境（`>= 8 GB` 显存，双门限要求 `nvidia-smi >= 5500 MiB` 与 `torch.cuda.mem_get_info >= 6144 MiB`）。

---

## 4. 历史工作归档（WACV 2019 11 类 / QLoRA R6-R7）

以下内容属于早期在 **WACV 2019 11 类 PCB 数据集** 上的基准探索、QLoRA 监督微调（R6 / R7）及旧版入口，与当前 FICS-PCB 6 类 8 板研究与最终演示主线完全解耦：
- 类别与任务旧配置：`configs/classes.yaml`、`configs/task.yaml`、`configs/lora*.yaml`
- 旧演示服务入口：`scripts/demo_app.py`、`scripts/fics_showcase.py`（分别面向早期 WACV 与 4 板回放）
- 旧批次评估与训练脚本：`scripts/train_lora.py`、`scripts/infer_batch.py`、`scripts/run_v2_comparison.py`、`scripts/run_v3_lora_comparison.py`、`scripts/train_yolo_v3.py`
