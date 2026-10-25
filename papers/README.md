# 毕业设计核心文献索引与导读指南

本项目聚焦于 **基于开源视觉语言大模型（VLM）的电子元器件检测**。本目录已下载并归档了 Codex 在前期论证中推荐的全部核心与拓展文献（共 9 篇全量 PDF），涵盖数据集规范、多模态大模型定位、密集小目标切片推理及参数高效微调。

---

## 一、 第一梯队：优先必读与精读文献（与课题直接对应）

| 文件名 | 论文名称与作者 | 重点看什么 | 读完需回答的核心问题 |
| :--- | :--- | :--- | :--- |
| [01_FICS-PCB_FPIC_Dataset_2022.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/01_FICS-PCB_FPIC_Dataset_2022.pdf) | **FPIC & FICS-PCB: A Novel Semantic Dataset for Optical PCB Assurance**<br>*Nathan Jessurun, Olivia Paradis, Mark Tehranipoor, Damon Woodard, Navid Asadizanjani et al. (2022)* | 数据采集方式、Microscope（显微镜微距）与 DSLR（单反大视野）两组子集、元件类别定义与标注方式 | 1. 检测哪几类元器件？<br>2. 为什么光照、缩放和传感器倍率会剧烈影响识别？<br>3. 为什么必须按物理板隔离防止数据泄露？ |
| [02_Qwen2.5-VL_2025.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/02_Qwen2.5-VL_2025.pdf) | **Qwen2.5-VL Technical Report**<br>*Shuai Bai et al. (2025)* | 动态分辨率 ViT 架构、图像定位（Grounding）语法、边界框归一化坐标规范、模型 SFT 样本格式 | 1. 高分辨率图片如何动态切成 Visual Tokens？<br>2. 为什么小元件需要保持长宽比输入？<br>3. 模型如何用文本输出坐标？ |
| [03_DeepSeek-VL2_2024.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/03_DeepSeek-VL2_2024.pdf) | **DeepSeek-VL2: Mixture-of-Experts Vision-Language Models**<br>*Zhiyu Wu et al. (2024)* | 动态分块视觉编码器、MoE 架构、原生 Grounding 标记格式（`<\|ref\|>` 与 `<\|det\|>`） | 1. 与 Qwen 的坐标输出协议有什么本质不同？<br>2. 为什么 MoE 的“总参数量”和“激活参数量”在计算显存时不能等同？ |
| [04_SAHI_2022.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/04_SAHI_2022.pdf) | **Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI)**<br>*Fatih Cagatay Akyon et al. (ICIP 2022)* | 图像重叠切片（Tiling）、局部推理、原图坐标逆变换、跨切片重复框合并（NMS） | 1. 为什么将大图切片能显著降低微小贴片元件的漏检率？<br>2. 跨边界被裁断的元器件如何合并？ |

---

## 二、 第二梯队：理论根基与扩展文献（按需查阅）

| 文件名 | 论文名称与作者 | 知识点与查阅时机 |
| :--- | :--- | :--- |
| [05_Qwen-VL_2023.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/05_Qwen-VL_2023.pdf) | **Qwen-VL: A Versatile Vision-Language Model**<br>*Jinze Bai et al. (2023)* | 对应任务书题目的初代 Qwen-VL，用于论文第一章“相关工作与技术背景”对比初代与最新版本的演化。 |
| [06_DeepSeek-VL_2024.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/06_DeepSeek-VL_2024.pdf) | **DeepSeek-VL: Towards Real-World Vision-Language Understanding**<br>*Haoyu Lu et al. (2024)* | 初代 DeepSeek 视觉模型架构，混合视觉编码器（SigLIP + SAM）原理。 |
| [07_Attention_Is_All_You_Need_2017.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/07_Attention_Is_All_You_Need_2017.pdf) | **Attention Is All You Need**<br>*Ashish Vaswani et al. (NeurIPS 2017)* | Transformer 奠基之作。看懂 Self-Attention、Scaled Dot-Product、Multi-Head 及位置编码原理，为毕业论文理论章节打底。 |
| [08_ViT_2020.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/08_ViT_2020.pdf) | **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**<br>*Alexey Dosovitskiy et al. (ICLR 2021)* | 解释“图片如何被切分为 Patch Token 并输入 Transformer”，理解 VLM 中视觉特征提取的核心。 |
| [09_LoRA_2021.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/09_LoRA_2021.pdf) | **LoRA: Low-Rank Adaptation of Large Language Models**<br>*Edward Hu et al. (ICLR 2022)* | 低秩适配微调。理解为什么冻结主干、仅训练低秩矩阵 $W_0 + B \times A$ 能将训练显存大幅压缩，指导阶段 13~14 的微调参数设置。 |
| [11_Hardware_Trust_PCB_Lu_2020.pdf](file:///C:/workspace/Electronic-Component-Detection-with-Open-Source-VLMs/papers/11_Hardware_Trust_PCB_Lu_2020.pdf) | **Hardware Trust and Assurance through Reverse Engineering: A Survey**<br>*Hangwei Lu et al. (2020)* | FICS-PCB 作者团队关于印刷电路板（PCB）硬件逆向、图像分析与机器视觉质检的长篇综述，极佳的论文绪论背景引用来源。 |

---

## 三、 Codex 推荐的论文精读五要素法

每读一篇论文，只需在笔记中提取并回答以下 5 个问题：
1. **研究什么问题？**（例如：PCB 工业场景下高倍率/多光照带来的泛化难题）
2. **输入和输出是什么？**（例如：输入高分辨率 PCB 图片，输出元器件类别与坐标文本序列）
3. **核心方法是什么？**（例如：动态切块、ViT 下采样、重叠切片 SAHI、LoRA 适配器）
4. **用什么数据和指标验证？**（例如：FICS-PCB 数据集、Precision/Recall/F1、mAP50）
5. **哪部分可以直接用于我的毕设？**（例如：借鉴 SAHI 的切片去重逻辑，借鉴 Qwen 的 Grounding 提示词模板）
