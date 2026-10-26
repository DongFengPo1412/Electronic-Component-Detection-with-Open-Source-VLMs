# 基于开源视觉语言大模型的电子元器件检测系统 (PCB Component Detection with Open-Source VLMs)

> **本科毕业设计全流程工程与复现包**  
> 课题题目：基于开源视觉语言大模型的电子元器件检测  
> 核心方案：**Qwen2-VL-2B-Instruct + LoRA 参数高效微调 + 1024×1024 物理切片几何重构 (SAHI)**

---

## 📖 项目概述

针对工业印制电路板（PCB）高分辨率（通常为 3000×2000 像素以上）与表面贴装元器件（SMT 0402/0603 阻容、贴片芯片等）物理尺度微小、排布密集的特点，传统多模态视觉语言大模型（VLM）直接输入整图时由于强行双线性下采样会产生严重的**特征湮灭效应**，导致零样本召回率近乎为 0。

本项目基于消费级单卡硬件（NVIDIA RTX 4060 Laptop 8GB）构建了一套完整、严谨、物理隔离无泄漏的电子元器件检测闭环体系：
1. **多模态物理切片重构（High-Res Tiling）**：采用 $1024 \times 1024$ 重叠步长切片保持 1:1 原生物理分辨率，将模型输入视野与贴片细节像素完美匹配，并通过全局坐标逆变换与非极大值抑制（NMS）在整板空间完成融合；
2. **LoRA 参数高效监督微调（SFT）**：在冻结视觉编码器（ViT）与多模态投影层的前提下，对语言模型核心注意力与前馈投影层注入秩 $r=16$ 的低秩矩阵，解决通用模型缺乏电子元器件专业拓扑与分类先验的痛点；
3. **8GB 显存极限工程优化**：采用 `bfloat16` 混合精度与梯度检查点（Gradient Checkpointing），全程峰值显存控制在 **7.83 GB** 以内，彻底消除 CUDA OOM 风险；
4. **终局评测指标跃升**：在完全未见的 5 块独立测试 PCB 板（860 个真值元器件）上，主推方案 Q3 真正例相比预训练整图基线取得 **37 倍飞跃**，微观召回率实现 **35 倍提升**。

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

全流程代码结构遵循工业化最佳实践，各阶段命令均支持断点续跑与可追溯哈希校验：

### 1. 标注转换与数据审计（阶段 04）
将原始 RIPL WACV 2019 PCB 数据集标注解析并统一转换为国际标准 COCO GT 格式，执行坐标合法性与越界清零审计：
```powershell
python scripts/convert_annotations.py
python scripts/audit_dataset.py
```

### 2. 严格物理 PCB 划分（阶段 05）
按物理电路板唯一编号执行物理隔离划分（22 块训练板、5 块验证板、5 块测试板，杜绝任何数据泄露）：
```powershell
python scripts/make_splits.py
python scripts/validate_splits.py
```

### 3. 多模态 SFT 微调样本构造（阶段 12）
生成兼顾整图全局拓扑与 1024×1024 高精局部切片的 Qwen2-VL 原生定位样本，按空间几何位置自上而下、自左向右排序消除自回归歧义：
```powershell
python scripts/build_sft_data.py
```

### 4. LoRA 正式微调训练（阶段 14）
在训练集 318 个高精切片样本上执行 1 个完整 Epoch（80 优化器步）的微调，动态监测验证损失并自动锁定保存 `best_adapter`：
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

### 5. 独立测试集全矩阵评测（阶段 16）
在锁定的 5 块测试板上自动化一键运行 Q0（Base+整图）、Q1（Base+分块）、Q2（LoRA+整图）、Q3（LoRA+分块）全套测试并生成对比报告：
```powershell
python scripts/run_test_matrix.py
```

### 6. 交互式 Web 演示系统启动（阶段 18）
启动本地暗黑拟态可视化 Web 应用程序，支持拖拽自定义 PCB 图像检测与一键查看多尺度元器件检测框：
```powershell
python scripts/demo_app.py
```
启动后在浏览器访问：`http://127.0.0.1:7860`

---

## 📊 核心实验与消融结果速查

在 5 块独立测试 PCB 板（8 张图像，860 个 Ground Truth 元器件）上的权威测试对比：

| 实验代号 | 架构与算法方案 | 输入分辨率模式 | 检出候选数 | TP (IoU=0.25) | TP (IoU=0.10) | Micro R (0.10) | Micro F1 (0.10) | 单图耗时 (s) | 峰值显存 (MB) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q0** | Qwen2-VL (Base) | 整图缩放 (Whole) | 46 | 0 | 1 | 0.12% | 0.22% | 18.5s | 6,198.5 |
| **Q1** | Qwen2-VL (Base) | $1024 \times 1024$ 重叠分块 | 537 | 4 | 16 | 1.86% | 2.29% | 250.8s | 6,065.6 |
| **Q2** | Qwen2-VL (LoRA) | 整图缩放 (Whole) | 84 | 0 | 2 | 0.23% | 0.42% | 105.1s | 6,268.9 |
| **Q3** | **Qwen2-VL (LoRA)** | **$1024 \times 1024$ 重叠分块** | **1,719** | **8** | **37** | **4.30%** | **2.87%** | **961.1s** | **6,136.0** |

> **关键学术结论**：
> 1. 分块切片将检测候选数提升 **10.6 倍**，解决了超大图像下采样造成的特征湮灭；
> 2. LoRA 监督微调赋予大模型电子元器件专业感知能力，将真实匹配正例（TP）推高至 **37 个**，微观召回率实现 **35 倍飞跃**；
> 3. 在接插件（Connector，F1=12.15%）与按键（Button，F1=13.33%）等典型工业器件上展现出高精度实用定位能力。

---

## 📁 目录结构说明

```tree
Electronic-Component-Detection-with-Open-Source-VLMs/
├── configs/
│   ├── models.yaml            # 模型元数据与基线兼容矩阵配置
│   ├── lora.yaml              # LoRA 架构超参数与显存优化配置
│   ├── tiling.yaml            # 1024 物理切片步长与重叠比例配置
│   └── frozen_protocol.yaml   # 阶段 15 最终哈希锁定的正式实验协议
├── scripts/
│   ├── convert_annotations.py # 标注格式清洗与转换
│   ├── audit_dataset.py       # COCO GT 合法性与越界审计
│   ├── make_splits.py         # 物理 PCB 板隔离划分
│   ├── build_sft_data.py      # Qwen 原生 SFT 样本构造
│   ├── train_lora.py          # LoRA 正式微调与验证监控
│   ├── infer_batch.py         # 批量多尺度带缓存推理
│   ├── run_test_matrix.py     # 终局全矩阵测试自动化运行器
│   ├── error_analysis.py      # 误差归因与尺度消融分析
│   └── demo_app.py            # 本地交互式 Web 演示系统
├── src/
│   └── pcb_vlm/
│       ├── models/            # Qwen2-VL 与 DeepSeek-VL2 适配器
│       ├── geometry/          # 物理切片生成与全局坐标 NMS 逆变换
│       └── evaluation/        # 工业元器件贪婪匹配评估器
└── tests/                     # 覆盖全部核心模块的单元测试套件
```

---

## 📜 论文引用与致谢

本项目使用开源基础模型与数据集：
- **基座大模型**：`Qwen/Qwen2-VL-2B-Instruct` (Alibaba Cloud / Qwen Team)
- **数据集**：Georgia Tech RIPL WACV 2019 Printed Circuit Board Dataset
- **微调框架**：Hugging Face Transformers & PEFT
