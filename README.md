# 基于开源视觉语言大模型的电子元器件检测实证研究 (Empirical Study on PCB Component Detection with Open-Source VLMs)

> **本科毕业设计全流程工程与复现包**  
> 课题题目：基于开源视觉语言大模型的电子元器件检测  
> 核心探索：**轻量多模态 VLM (Qwen2-VL-2B / DeepSeek-VL-1.3B) 在密集电子元器件检测中的定位能力边界、消融实验与失效机理分析**

---

## 📖 项目概述与实证结论

针对工业印制电路板（PCB）高分辨率（通常为 3000×2000 像素以上）与表面贴装元器件（SMT 0402/0603 阻容、贴片芯片等）物理尺度微小、排布密集的特点，通用多模态视觉语言大模型（VLM）在整图输入与微调检测中面临严峻挑战。

本项目在单卡消费级硬件（NVIDIA RTX 4060 Laptop 8GB）上构建了一套可复现、严格板级隔离的开源 VLM 检测与微调评测平台，主要研究结论如下：

1. **轻量 VLM 检测能力边界明确**：
   - **DeepSeek-VL-1.3B-chat** 定位为通用视觉语言对话基座，生成自然语言描述，无原生定位标记输出，在数值坐标检测评测中检出框为 0；
   - **Qwen2-VL-2B-Instruct** 在整图缩放输入下，由于微小器件在下采样后分辨率急剧丢失（低于 Patch 分辨率极限），基础模型与微调模型检出率均极低（标准 IoU≥0.50 下 TP=0）。
2. **分块（Tiling）与微调的收益与失效机理**：
   - 引入高分辨率分块切片后，候选生成框数量显著增加（从数十个增至千余个）；
   - 但在标准检测指标（IoU≥0.50）下，LoRA 微调（Q3）仅取得 1 个真正例（TP），低于基础模型分块（Q1）的 2 个 TP，微调尚未形成稳定正收益；
   - 早期实验所观察到的“低门槛（IoU≥0.10）TP 增长”（从 1 个增至 37 个），伴随产生超过 1,600 个沿 PCB 铜线走线分布的阶梯状虚警框（Ladder Hallucination），本质属于语言模型自回归重复生成的结构性失效，不代表工业高精度定位能力。
3. **数据集与板级隔离规范**：
   - 实验采用 **Georgia Tech RIPL WACV 2019 PCB 数据集**（共 47 张高分辨率图像，对应 29 块独立物理板），选取 11 类元器件；该数据集需与 FICS-PCB / FPIC（显微镜/DSLR 子集）严格区分；
   - 建立了经人工复核的物理板映射规范（`configs/board_canonical_map.json`），将多视图（如 `ML450_Top1/Top2/Bottom1/Bottom2`、`XLVDSproSupply`）统一归入单一物理板，彻底杜绝板级跨集合泄漏。

---

## 🛠️ 软硬件环境要求

- **操作系统**：Windows 11 64-bit / Linux x86_64
- **显卡配置**：NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) 或更高规格显卡
- **显卡驱动与 CUDA**：Driver $\ge 560.94$，CUDA 12.1+
- **Python 环境**：Python 3.10 / 3.11（推荐 Anaconda 环境 `ai_cv_env`）
- **核心依赖包**：
  ```bash
  torch==2.5.1+cu121
  transformers>=4.45.0
  peft>=0.13.0
  qwen-vl-utils
  pycocotools
  pillow
  pyyaml
  ```

---

## 🚀 核心工作流与复现命令指南

### 1. 规范化物理板映射与标注转换
使用规范物理板映射表转换并切分数据集：
```powershell
python scripts/convert_annotations.py
python scripts/make_splits.py --seed 42
python scripts/validate_splits.py
```

### 2. 坐标与模型协议审计
验证非正方形图像坐标轴映射及模型定位协议：
```powershell
python scripts/verify_coordinate_protocol.py
python -m unittest tests/test_qwen_adapter.py
```

### 3. 多模态 LoRA 微调与全量验证评估
支持全验证集损失评估与优化器断点状态保存：
```powershell
python scripts/train_lora.py `
  --model-path "C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master" `
  --data-file "C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\sft_data\train_sft.json" `
  --val-data-file "C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\sft_data\val_sft.json" `
  --output-dir "C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage14_lora_training" `
  --config "configs/lora.yaml" `
  --max-steps 80 `
  --eval-steps 20
```

### 4. 标准多阈值检测评测
主报标准检测指标（IoU 0.50, 0.75, AP@[.50:.95]），辅助诊断指标严格标记：
```powershell
python scripts/evaluate.py `
  --gt "C:/Users/Liu/.codex/cache/tasks/pcb-vlm-antigravity-20261024/annotations/coco_annotations.json" `
  --pred "C:/Users/Liu/.codex/cache/tasks/pcb-vlm-antigravity-20261024/runs/stage16_final_test_matrix/Q3_qwen_lora_tiling/predictions.jsonl" `
  --benchmark
```

### 5. 交互式真实回放演示系统
启动本地 Web 应用程序，严格按磁盘真实推理文件回放，不造假、不填充伪造框：
```powershell
python scripts/demo_app.py
```
访问：`http://127.0.0.1:7860`

---

## 📊 历史探索结果（旧划分：8 张图像、860 个 GT；非当前正式测试）

| 方案代号 | 模型架构与配置 | 输入形式 | 预测框总数 | 标准 TP (IoU≥0.50) | 标准 Precision (0.50) | 标准 Recall (0.50) | 诊断 TP (IoU≥0.10) | 诊断 Precision (0.10) | 单图耗时 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q0** | Qwen2-VL-2B (Base) | 整图缩放 (Whole) | 46 | **0** | 0.00% | 0.00% | 1 | 2.17% | ~18s |
| **Q1** | Qwen2-VL-2B (Base) | 1024 分块 (Tiling) | 537 | **2** | 0.37% | 0.23% | 16 | 2.98% | ~251s |
| **Q2** | Qwen2-VL-2B (LoRA) | 整图缩放 (Whole) | 84 | **0** | 0.00% | 0.00% | 2 | 2.38% | ~105s |
| **Q3** | Qwen2-VL-2B (LoRA) | 1024 分块 (Tiling) | 1,719 | **1** | 0.06% | 0.12% | 37 | 2.15% | ~961s |
| **DS** | DeepSeek-VL-1.3B | 对话描述 (Whole) | 0 | **0** | 0.00% | 0.00% | 0 | 0.00% | ~7s |

> **实验客观诊断分析**：
> 1. **标准评价结论**：在目标检测通行评价标准（IoU≥0.50 / 0.75）下，当前轻量 VLM 无法稳定解决密集微小元器件的精确定位；微调方案 Q3 的标准匹配数（1 个）并未优于基础模型 Q1（2 个）；
> 2. **宽松门槛归因**：IoU≥0.10 下所谓 37 个 TP 伴随 1,682 个误报，且误报框呈现沿 PCB 铜线走线的连续阶梯状排列，属于自回归语言生成的幻觉与漂移；
> 3. **算力与工程意义**：8GB 显卡下单图分块耗时约 16 分钟，工业落地需要结合专用检测先验（如轻量级局部特征注入或两阶段级联架构），而非单纯依靠通用自回归 VLM 端到端输出坐标。

---

## 📁 目录结构说明

```tree
Electronic-Component-Detection-with-Open-Source-VLMs/
├── configs/
│   ├── board_canonical_map.json # 29 块物理板规范映射字典（杜绝多视角泄漏）
│   ├── models.yaml              # 模型基座与配置定义
│   ├── lora.yaml                # LoRA 结构超参数与显存优化配置
│   └── tiling.yaml              # 1024 物理切片步长与重叠比例配置
├── scripts/
│   ├── convert_annotations.py   # 规范物理板映射与标注转换
│   ├── make_splits.py           # 物理 PCB 板级隔离划分
│   ├── validate_splits.py       # 严格物理跨集重叠与图片隔离校验
│   ├── verify_coordinate_protocol.py # 坐标轴与模型格式协议验证
│   ├── train_lora.py            # LoRA 微调训练与全量验证监控
│   ├── infer_batch.py           # 批量推理（含哈希缓存、原始文本保存与耗时纠正）
│   ├── evaluate.py              # 标准多阈值检测评估器（IoU 0.50/0.75/诊断 0.10）
│   └── demo_app.py              # Web 演示系统（真实推理回放，无任何伪造数据）
├── src/
│   └── pcb_vlm/
│       ├── models/              # Qwen2-VL 与 DeepSeek-VL-1.3B 适配器
│       ├── geometry/            # 几何切片生成与坐标映射变换
│       └── evaluation/          # 目标检测匹配算法
└── tests/                       # 单元测试与坐标协议验证套件
```

---

## 📜 数据集与参考文献规范

- **基座大模型**：
  - `Qwen/Qwen2-VL-2B-Instruct` (Alibaba Cloud / Qwen Team)
  - `deepseek-ai/deepseek-vl-1.3b-chat` (DeepSeek)
- **实验数据集**：
  - Georgia Tech RIPL Printed Circuit Board Dataset: Kuo et al., "Expert-Level Detection of PCB Components with Deep Learning", WACV 2019. (47 high-res PCB images, 29 canonical physical boards).
- **特别声明**：
  - 本项目当前实验所用数据为上述 WACV 2019 数据集，非 FICS-PCB / FPIC 数据集；两者拍摄设备与标注规范不同，在毕业论文中须明确区分并向导师报备确认。



### P0 实验边界（2026-11-04）
历史 8 图 / 860 GT、旧划分 LoRA、外部探索实验均不等于当前冻结协议的正式测试结果。
正式测试以 canonical splits.json 的 MicroZed、Spartan3、ThingMagicM6e、Virtex6（5 图）为准；不得为复用旧成绩更改测试集。
旧 SFT 缺少逐条来源字段、旧 LoRA 缺少完整元数据时保留原文件，仅供历史诊断；新的正式入口会拒绝这些输入。
正式推理参数从 configs/frozen_protocol.yaml 读取；变更参数须另存新协议并指定 --protocol。
COCO AP 默认报告 maxDets=[1,10,100]；evaluation.yaml 的密集扩展设置独立列于 dense_extended。
无置信度 VLM 使用等分数排序，其 AP 为补充指标，不是校准置信度。主要报告 IoU=0.50 的 P/R/F1、TP/FP/FN 与失败状态。


### P1 坐标与模型适配边界（2026-11-04）
正式 Qwen 解析使用 `xyxy_norm1000`，不再按数字大小猜测尺度，也不交换倒置坐标或裁剪非法坐标。
DeepSeek-VL-1.3B-chat 使用项目明确要求的 `yxyx_norm1000` 数值回答协议；这不是已验证的原生 grounding 能力，正式入口拒绝 VL2 的 `<|ref|>/<|det|>` 标记。
共享几何解析器还可显式指定 norm1 或 pixels，以供对应输入协议的诊断；这些机械转换测试不证明模型支持所有格式。
历史启发式解析必须显式 `legacy=True`，审计状态标记为非正式兼容。`verify_coordinate_protocol.py` 是历史原文回放，不是新 GPU 推理。
缺失/未知类别不会变成合法元器件，过小或细长但几何合法的框不再被固定阈值误删；重复框保留并计数，交由声明的 NMS/评估处理。
每次预测保存原文、坐标协议、解析各阶段计数、token/EOS 与截断状态。DeepSeek v1 的图像处理按自身 processor 执行固定 resize/pad，`max_pixels` 不作为实际生效参数。
官方依据：[Qwen2-VL 论文 2.2.1](https://arxiv.org/html/2409.12191v2#S2.SS2.SSS1)、[DeepSeek-VL v1](https://github.com/deepseek-ai/DeepSeek-VL)、[DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)。


### 验证数据语义警报（2026-11-04）
原始XML包含 `text "LED4/P1.21"`、`text IC1` 等文字标签。旧转换器按整条字符串匹配类别，导致现有canonical标注中至少170条文字框被归入LED/IC；哈希及几何审计通过并不证明类别语义正确。
转换器现只匹配XML类别字段；audit_dataset增加原XML逐框语义审计。旧数据和历史权重保留，但相关训练/评估须按受污染历史实验理解，不能直接作为正确元器件检测的正式证据。
本轮诊断使用单独的语义纠错验证真值副本（716GT），没有覆盖原740GT或变更测试板划分，不能与旧成绩直接比较。


### 语义规范数据 V3（2026-11-04）
从全部18201个XML对象重新建立V3，保留6594个目标，不以6478为预设数量。旧版漏掉116个合法目标（108 connector、5 clock、button/switch/inductor各1）。对账：6476 unchanged、178 removed、2 relabeled、116 added；每个原对象都有完整ledger记录。
新数据与旧数据完全分开：`C:/Users/Liu/.codex/cache/tasks/pcb-v3-20261104/annotations_v3_semantic`；SFT为同级 `sft_v3_semantic`。旧47图身份与35/7/5图、21/4/4板划分保持；新GT数量为5044/759/791，旧740验证GT与716局部纠错副本均不等于V3完整验证集。
使用 `scripts/rebuild_semantic_dataset.py` 及 `configs/semantic_policy_v3.yaml` 重建，`scripts/audit_semantic_v3.py` 独立检查全部XML双向覆盖和SFT像素/监督来源；已有输出目录不会覆盖。
新Train/Val SFT为451/71条，不生成Test SFT；测试集修正连接器仍只在测试GT中。新的冻结协议另存数据目录的 `frozen_protocol_v3.yaml`，旧冻结协议不覆盖；正式推理须显式 `--protocol <V3协议>`，训练须显式 `--splits-path <V3划分>` 与新SFT路径。
注意训练87条、验证17条回答超过1024 tokens，均保留完整监督而没有截断；数据审计通过不意味着整套样本可直接在8GB显卡上训练。下一轮必须明确长度预算与样本覆盖。本轮不训练。


### V3 训练准备与工程试运行（2026-11-04）
本地数据入口见 `configs/data_v3.yaml`，包括原V3、派生SFT和经过逐文件SHA-256核验的长期恢复清单。快照保留原路径引用；恢复应按清单复制缺失文件并核验，遇冲突停止，不移动/删除缓存。原V3完整SFT保持不变。
`scripts/audit_training_budget.py --sft <目录> --model <本地Qwen目录> --output <新的任务缓存目录> --sequence-budget 1536` 使用实际processor/collator统计完整输入、完整回答与覆盖，不加载模型权重。不要把推理max_new_tokens当作训练长度。
新增512空间切片后，保留完整回答且联合长度≤1536的派生集合为Train 1637条/4525唯一目标，Val 237条/650唯一目标；覆盖21/4物理板和全部11类。它不是完整V3监督：训练仍缺519个、验证缺109个唯一目标。原V3总量仍为5044/759/791。
`configs/lora_v3_smoke.yaml` 仅用于工程验证：NF4 4-bit、microbatch=1、梯度累积2、总12更新步（先停10，再从checkpoint-10恢复至12）；`--max-steps` 是固定调度器总步数，`--stop-after-step` 仅提前停止，不改变学习率曲线。输出目录须新建，恢复从旧目录读、向新目录写。
实际24条Train、4条Val试运行完成12步，验证了反向传播、LoRA更新和检查点恢复，分配显存峰值6244.60 MiB。此结果不证明检测质量改善，也不证明全量/长序列训练安全；显存余量很小。源程序随后修正恢复时将final误标best的后备逻辑，并通过微型真实训练循环回归；旧试运行误命名目录已标记，权重未改写。最后权重不再自动提升为best。
本机共享环境huggingface-hub 2.4.0与transformers 4.46.3不兼容。本轮只在任务缓存dependency-overlay安装hub 0.36.2，并通过PYTHONPATH使用；没有修改共享环境。QLoRA实际版本：torch 2.5.1+cu121、transformers 4.46.3、peft 0.21.2、bitsandbytes 0.50.2。后续执行前应检查环境，不应默认为全局环境已修好。
完整结果和运行命令见 `C:/Users/Liu/.codex/artifacts/pcb-train-prep-20261104/执行报告.md`；不使用测试集、没有新检测性能成绩、没有推送远程。


### V3 正式基线对照（R6 有界 QLoRA 与 R7 YOLO11n，2026-11-04 ～ 2026-11-12）
在完整 V3 验证集（7 张图像、21→4 物理板严格隔离、共 **759 个真值目标**；冻结 Test 集 5 图 / 791 GT 全程未触碰）下的最新基线结果：

| 轮次与方案 | 口径 / 配置 | TP | FP | FN | Precision | Recall | F1 | AP50 (std) | mAP50:95 (std) | 类无关 IoU@0.5 召回 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **R6 `B0`** (`Qwen2-VL-2B-Instruct` Base) | V3 Val 全图 1024 切片（68 块，NF4，256 token 上限） | 1 | 576 | 758 | 0.0017 | 0.0013 | 0.0015 | — | — | `1/759` (0.13%) |
| **R6 `L40`** (`v3-qlora-r6-bounded160`, step 40) | 同上（0.195 Epoch / 160 步有界训练中精选检查点） | 0 | 76 | 759 | 0.0000 | 0.0000 | 0.0000 | — | — | `1/759` (0.13%) |
| **R7 `Y0`** (`YOLO11n` Best Epoch 53，整图) | V3 Val 整图推理（`imgsz=640`，主指标 `conf=0.25`） | 124 | 158 | 635 | 0.4397 | 0.1634 | 0.2382 | 0.1328 | 0.0769 | `166/759` (21.87%) |
| **R7 `Y1`** (`YOLO11n` Best Epoch 53，切片) | V3 Val 640×640 重叠 0.2 切片（160 次前向） | **492** | 462 | 267 | **0.5157** | **0.6482** | **0.5744** | **0.3277** | **0.2001** | **`565/759` (74.44%)** |

> [!NOTE]
> 1. **非单变量架构对比声明**：R6（`B0`/`L40`）与 R7（`Y0`/`Y1`）在预训练任务、输入分辨率、训练轮次（0.195 epoch vs 60 epochs）与置信度校准上不同，不应作为纯架构因果对照，各自定位为 V3 验证集上的 VLM 参考基线与专用轻量检测基线。
> 2. **跨机路径覆盖与依赖**：适配器支持通过环境变量 `PCB_VLM_CACHE_ROOT`、`PCB_VLM_QWEN_PATH`、`PCB_VLM_DEEPSEEK_PATH` 覆盖默认本地模型快照路径；当前 `scripts/demo_app.py` 仍展示历史旧缓存演示，尚未迁移至 V3 / R7。
> 3. **许可证与第三方组件边界**：R7 专用检测基线使用 `ultralytics`（`AGPL-3.0`，见 `configs/yolo_v3.yaml`），开源权重 `Qwen2-VL-2B-Instruct` 与 `DeepSeek-VL-1.3B-chat` 须分别遵循各自模型许可条款；公开分发时应明确区分学术研究代码与第三方权重/依赖许可边界。
