# FICS-PCB 6 类检测主实验自动汇总表（Machine-Generated）

> 本文档由 `scripts/build_fics_results_bundle.py` 直接读取已校验 SHA-256 的正式回放与会话遥测 JSON/JSONL 自动生成，未在脚本中手写任何实验分数。
> 两个样本集（`s18+s28` 开发集 64 GT 与 `s29+s27` 未观察新板 41 GT）严格分开呈现，不跨数据集混合求平均。

## 1. 统一实用预算（512 tokens）三模型主结果对比

### FICS-PCB Dev Two-Board (s18_front + s28_front)（201 次推理，每模型 67 次，共 64 GT）

| 模型 | 模式 | TP | FP | FN | Pred | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-VL-2B-Instruct | W (整 ROI) | 0 | 11 | 64 | 11 | 0.00% | 0.00% | 0.00% |
| Qwen2-VL-2B-Instruct | T (切片+NMS) | 0 | 355 | 64 | 355 | 0.00% | 0.00% | 0.00% |
| deepseek-vl-1.3b-chat | W (整 ROI) | 0 | 0 | 64 | 0 | 0.00% | 0.00% | 0.00% |
| deepseek-vl-1.3b-chat | T (切片+NMS) | 0 | 6 | 64 | 6 | 0.00% | 0.00% | 0.00% |
| Qwen3-VL-2B-Instruct | W (整 ROI) | 3 | 13 | 61 | 16 | 18.75% | 4.69% | 7.50% |
| Qwen3-VL-2B-Instruct | T (切片+NMS) | 19 | 257 | 45 | 276 | 6.88% | 29.69% | 11.18% |

### FICS-PCB Unobserved Two-Board (s29 + s27)（306 次推理，每模型 102 次，共 41 GT）

| 模型 | 模式 | TP | FP | FN | Pred | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-VL-2B-Instruct | W (整 ROI) | 0 | 10 | 41 | 10 | 0.00% | 0.00% | 0.00% |
| Qwen2-VL-2B-Instruct | T (切片+NMS) | 1 | 550 | 40 | 551 | 0.18% | 2.44% | 0.34% |
| deepseek-vl-1.3b-chat | W (整 ROI) | 0 | 0 | 41 | 0 | 0.00% | 0.00% | 0.00% |
| deepseek-vl-1.3b-chat | T (切片+NMS) | 0 | 0 | 41 | 0 | 0.00% | 0.00% | 0.00% |
| Qwen3-VL-2B-Instruct | W (整 ROI) | 6 | 16 | 35 | 22 | 27.27% | 14.63% | 19.05% |
| Qwen3-VL-2B-Instruct | T (切片+NMS) | 12 | 235 | 29 | 247 | 4.86% | 29.27% | 8.33% |

## 2. 逐板拆分表（IoU >= 0.50，6 类类感知匹配）

| 数据集 | 模型 | 板卡 | 模式 | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev_two_board_s18_s28 | Qwen2-VL-2B-Instruct | s18_front | W | 36 | 0 | 6 | 36 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | Qwen2-VL-2B-Instruct | s18_front | T | 36 | 0 | 88 | 36 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | Qwen2-VL-2B-Instruct | s28_front | W | 28 | 0 | 5 | 28 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | Qwen2-VL-2B-Instruct | s28_front | T | 28 | 0 | 267 | 28 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | deepseek-vl-1.3b-chat | s18_front | W | 36 | 0 | 0 | 36 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | deepseek-vl-1.3b-chat | s18_front | T | 36 | 0 | 0 | 36 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | deepseek-vl-1.3b-chat | s28_front | W | 28 | 0 | 0 | 28 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | deepseek-vl-1.3b-chat | s28_front | T | 28 | 0 | 6 | 28 | 0.00% | 0.00% | 0.00% |
| dev_two_board_s18_s28 | Qwen3-VL-2B-Instruct | s18_front | W | 36 | 1 | 6 | 35 | 14.29% | 2.78% | 4.65% |
| dev_two_board_s18_s28 | Qwen3-VL-2B-Instruct | s18_front | T | 36 | 12 | 122 | 24 | 8.96% | 33.33% | 14.12% |
| dev_two_board_s18_s28 | Qwen3-VL-2B-Instruct | s28_front | W | 28 | 2 | 7 | 26 | 22.22% | 7.14% | 10.81% |
| dev_two_board_s18_s28 | Qwen3-VL-2B-Instruct | s28_front | T | 28 | 7 | 135 | 21 | 4.93% | 25.00% | 8.24% |
| unobserved_two_board_s29_s27 | Qwen2-VL-2B-Instruct | s29 | W | 19 | 0 | 5 | 19 | 0.00% | 0.00% | 0.00% |
| unobserved_two_board_s29_s27 | Qwen2-VL-2B-Instruct | s29 | T | 19 | 1 | 268 | 18 | 0.37% | 5.26% | 0.69% |
| unobserved_two_board_s29_s27 | Qwen2-VL-2B-Instruct | s27 | W | 22 | 0 | 5 | 22 | 0.00% | 0.00% | 0.00% |
| unobserved_two_board_s29_s27 | Qwen2-VL-2B-Instruct | s27 | T | 22 | 0 | 282 | 22 | 0.00% | 0.00% | 0.00% |
| unobserved_two_board_s29_s27 | deepseek-vl-1.3b-chat | s29 | W | 19 | 0 | 0 | 19 | 0.00% | 0.00% | 0.00% |
| unobserved_two_board_s29_s27 | deepseek-vl-1.3b-chat | s29 | T | 19 | 0 | 0 | 19 | 0.00% | 0.00% | 0.00% |
| unobserved_two_board_s29_s27 | deepseek-vl-1.3b-chat | s27 | W | 22 | 0 | 0 | 22 | 0.00% | 0.00% | 0.00% |
| unobserved_two_board_s29_s27 | deepseek-vl-1.3b-chat | s27 | T | 22 | 0 | 0 | 22 | 0.00% | 0.00% | 0.00% |
| unobserved_two_board_s29_s27 | Qwen3-VL-2B-Instruct | s29 | W | 19 | 3 | 11 | 16 | 21.43% | 15.79% | 18.18% |
| unobserved_two_board_s29_s27 | Qwen3-VL-2B-Instruct | s29 | T | 19 | 7 | 110 | 12 | 5.98% | 36.84% | 10.29% |
| unobserved_two_board_s29_s27 | Qwen3-VL-2B-Instruct | s27 | W | 22 | 3 | 5 | 19 | 37.50% | 13.64% | 20.00% |
| unobserved_two_board_s29_s27 | Qwen3-VL-2B-Instruct | s27 | T | 22 | 5 | 125 | 17 | 3.85% | 22.73% | 6.58% |

## 3. Qwen3-VL-2B-Instruct 逐类拆分（含 GT=0 零分母规范标注）

> 说明：在 `s29+s27` 样本中，`diode` 的真实 GT 支持数为 `0`；按照严格定义，其 Recall 与 F1 标注为 `N/A (GT=0)`，误报数 `FP` 照实列出。

### FICS-PCB Dev Two-Board (s18_front + s28_front) — Qwen3-VL-2B-Instruct

| 模式 | 类别 | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W | `capacitor` | 30 | 4 | 1 | 3 | 29 | 25.00% | 3.33% | 5.88% |
| W | `resistor` | 21 | 3 | 0 | 3 | 21 | 0.00% | 0.00% | 0.00% |
| W | `inductor` | 4 | 1 | 0 | 1 | 4 | 0.00% | 0.00% | 0.00% |
| W | `transistor` | 1 | 1 | 0 | 1 | 1 | 0.00% | 0.00% | 0.00% |
| W | `diode` | 2 | 4 | 0 | 4 | 2 | 0.00% | 0.00% | 0.00% |
| W | `ic` | 6 | 3 | 2 | 1 | 4 | 66.67% | 33.33% | 44.44% |
| T | `capacitor` | 30 | 115 | 11 | 104 | 19 | 9.57% | 36.67% | 15.17% |
| T | `resistor` | 21 | 111 | 5 | 106 | 16 | 4.50% | 23.81% | 7.58% |
| T | `inductor` | 4 | 6 | 0 | 6 | 4 | 0.00% | 0.00% | 0.00% |
| T | `transistor` | 1 | 13 | 0 | 13 | 1 | 0.00% | 0.00% | 0.00% |
| T | `diode` | 2 | 9 | 1 | 8 | 1 | 11.11% | 50.00% | 18.18% |
| T | `ic` | 6 | 22 | 2 | 20 | 4 | 9.09% | 33.33% | 14.29% |

### FICS-PCB Unobserved Two-Board (s29 + s27) — Qwen3-VL-2B-Instruct

| 模式 | 类别 | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W | `capacitor` | 18 | 5 | 4 | 1 | 14 | 80.00% | 22.22% | 34.78% |
| W | `resistor` | 11 | 13 | 0 | 13 | 11 | 0.00% | 0.00% | 0.00% |
| W | `inductor` | 4 | 1 | 0 | 1 | 4 | 0.00% | 0.00% | 0.00% |
| W | `transistor` | 2 | 1 | 0 | 1 | 2 | 0.00% | 0.00% | 0.00% |
| W | `diode` | 0 | 0 | 0 | 0 | 0 | 0.00% | N/A (GT=0) | N/A (GT=0) |
| W | `ic` | 6 | 2 | 2 | 0 | 4 | 100.00% | 33.33% | 50.00% |
| T | `capacitor` | 18 | 71 | 5 | 66 | 13 | 7.04% | 27.78% | 11.24% |
| T | `resistor` | 11 | 94 | 2 | 92 | 9 | 2.13% | 18.18% | 3.81% |
| T | `inductor` | 4 | 17 | 0 | 17 | 4 | 0.00% | 0.00% | 0.00% |
| T | `transistor` | 2 | 17 | 2 | 15 | 0 | 11.76% | 100.00% | 21.05% |
| T | `diode` | 0 | 23 | 0 | 23 | 0 | 0.00% | N/A (GT=0) | N/A (GT=0) |
| T | `ic` | 6 | 25 | 3 | 22 | 3 | 12.00% | 50.00% | 19.35% |

## 4. `v2` 终止语义与格式遵从性统计

| 数据集 | 模型 | 规划/完成调用 | 正常 EOS | 达上限无 EOS | EOS 后格式/模式异常 | 严格 Schema 通过 | 兼容部分恢复 | 显式空数组 `[]` | 无框失败 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev_two_board_s18_s28 | Qwen2-VL-2B-Instruct | 67/67 | 55 | 12 | 7 | 48 | 18 | 0 | 1 |
| dev_two_board_s18_s28 | deepseek-vl-1.3b-chat | 67/67 | 59 | 8 | 58 | 1 | 0 | 0 | 66 |
| dev_two_board_s18_s28 | Qwen3-VL-2B-Instruct | 67/67 | 57 | 10 | 0 | 57 | 10 | 21 | 0 |
| unobserved_two_board_s29_s27 | Qwen2-VL-2B-Instruct | 102/102 | 92 | 10 | 16 | 76 | 26 | 0 | 0 |
| unobserved_two_board_s29_s27 | deepseek-vl-1.3b-chat | 102/102 | 93 | 9 | 93 | 0 | 0 | 0 | 102 |
| unobserved_two_board_s29_s27 | Qwen3-VL-2B-Instruct | 102/102 | 93 | 9 | 1 | 92 | 10 | 42 | 0 |

## 5. 时延、生成吞吐与 GPU 显存占用（来自实际原始 JSONL 与会话日志）

| 数据集 | 模型 | 调用数 | 阶段总耗时 (s) | 模型加载 (s) | 平均 GPU 生成 (ms) | W/T 平均生成 (ms) | 最大生成 (ms) | 平均预处理/H2D (ms) | 平均输出 Token | 吞吐 (tok/s) | 峰值分配/保留显存 (MiB) |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| dev_two_board_s18_s28 | Qwen2-VL-2B-Instruct | 67 | 746.48 | 9.79 | 10756.5 | 9812.6 / 10785.5 | 22781.6 | 28.6 / 1.71 | 263.2 | 24.5 | 4481.1 / 5040.0 |
| dev_two_board_s18_s28 | deepseek-vl-1.3b-chat | 67 | 348.17 | 8.86 | 4767.4 | 4748.7 / 4768.0 | 14676.7 | 119.2 / 0.90 | 173.7 | 36.4 | 4024.6 / 4248.0 |
| dev_two_board_s18_s28 | Qwen3-VL-2B-Instruct | 67 | 554.38 | 9.64 | 7904.3 | 13240.0 / 7740.1 | 24339.2 | 19.1 / 1.17 | 172.1 | 21.8 | 4193.9 / 4286.0 |
| unobserved_two_board_s29_s27 | Qwen2-VL-2B-Instruct | 102 | 899.79 | 9.34 | 8543.5 | 7532.3 / 8563.8 | 18579.2 | 26.3 / 1.56 | 238.7 | 27.9 | 4481.1 / 5040.0 |
| unobserved_two_board_s29_s27 | deepseek-vl-1.3b-chat | 102 | 434.48 | 9.34 | 3887.7 | 3972.5 / 3886.0 | 14261.7 | 121.9 / 0.95 | 139.4 | 35.9 | 4024.6 / 4248.0 |
| unobserved_two_board_s29_s27 | Qwen3-VL-2B-Instruct | 102 | 559.76 | 10.74 | 5182.3 | 20996.6 / 4866.1 | 27656.2 | 19.4 / 1.29 | 101.6 | 19.6 | 4193.9 / 4286.0 |

## 6. 单独补充：Qwen3 高预算开发集参考（`s18+s28`，不与 512-token 主表混排）

- 预算设置：`max_new_tokens=1024`，`qwen_max_pixels=1310720`（仅开发集参考；主线三模型对比严格以 512-token 同协议为准）

| 模式 | TP | FP | FN | Precision | Recall | F1 | 截断调用数 | 平均生成耗时 (ms) | 峰值分配显存 (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W | 4 | 54 | 60 | 6.90% | 6.25% | 6.56% | 2/2 | 60968.5 | 4712.3 |
| T | 35 | 269 | 29 | 11.51% | 54.69% | 19.02% | 11/65 | 18118.9 | 4555.1 |

## 7. 数据溯源 SHA-256 绑定清单

| 数据集 / 补充 | 构件 | SHA-256 (前 16 位) |
|---|---|---|
| dev_two_board_s18_s28 | `frozen_plan` | `790ec5e0905de8fe` |
| dev_two_board_s18_s28 | `gt_coco` | `dbde6f715e2f523e` |
| dev_two_board_s18_s28 | `replay_report` | `09683ff28ed82adc` |
| dev_two_board_s18_s28 | `independent_replay` | `064c6132c5ea0869` |
| dev_two_board_s18_s28 | `matrix_session_summary` | `1892f0be8274734d` |
| dev_two_board_s18_s28 | `raw_qwen2` | `babfe5f70b6b62a8` |
| dev_two_board_s18_s28 | `raw_deepseek` | `91efaf20389ccf80` |
| dev_two_board_s18_s28 | `raw_qwen3` | `90bdc54bb62a72b8` |
| unobserved_two_board_s29_s27 | `frozen_plan` | `bc462675f5e3cb43` |
| unobserved_two_board_s29_s27 | `replay_manifest` | `26354e56cb69866b` |
| unobserved_two_board_s29_s27 | `gt_coco` | `c336cc2e8a1aa280` |
| unobserved_two_board_s29_s27 | `replay_report` | `6333adcd66275545` |
| unobserved_two_board_s29_s27 | `independent_replay` | `47549dcb8e99007f` |
| unobserved_two_board_s29_s27 | `matrix_session_summary` | `79cc55ee881684d8` |
| unobserved_two_board_s29_s27 | `analysis_revision_manifest` | `c9396f6c3c2b7c3a` |
| unobserved_two_board_s29_s27 | `raw_qwen2` | `b6acf842b62f7ca2` |
| unobserved_two_board_s29_s27 | `raw_deepseek` | `e21ea0647c176c82` |
| unobserved_two_board_s29_s27 | `raw_qwen3` | `b3a8fd3522187113` |
