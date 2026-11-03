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
   - 实验采用 **Georgia Tech RIPL WACV 2019 PCB 数据集**（共 47 张高分辨率图像，对应 29 块独立物理板），选取 9 类主要元器件；该数据集需与 FICS-PCB / FPIC（显微镜/DSLR 子集）严格区分；
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

## 📊 实验实测结果对比（测试集 8 张图像，860 个 Ground Truth 元器件）

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

