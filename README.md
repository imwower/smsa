# SMSA — Self-Modeling Spiking Agent (标准库版)

> 在**仅使用 Python 标准库与普通 CPU**的条件下，实现：脉冲神经网络 (SNN) 在线学习（e-prop 三因子）、主动探索（内在动机）、自我模型 (Self-Model)、自调参与微演化（UCB + A/B 回滚），以及可选的字符级外部语料学习。

---

## 目录结构

```text
.
├─ envs/          # 环境
├─ meta/          # 元控制（自调参 / A/B / 回滚 / 热补丁）
├─ scripts/       # 训练入口脚本
├─ snn/           # SNN 核心模块
├─ tests/         # 最小单元测试
├─ tools/         # 日志工具（标准库）
├─ data/          # 外部语料或样例数据（可选）
└─ runs/          # 训练输出（CSV 等，已在 .gitignore）
```

**关键文件：**

- `envs/gridworld.py`：5×5 方格世界（上 / 下 / 左 / 右 + 随机滑移、逐步负奖、终点正奖）。
- `snn/lif.py`：LIF / ALIF 神经元（替代导数：三角 / fast-sigmoid / 可热补丁）。
- `snn/dense.py`：`DenseLIF` 网络 + e-prop 资格迹（`E_ij ← λ E_ij + ψ_j * pre_i`）。
- `snn/policy.py`：策略头（softmax 采样 + REINFORCE）。
- `snn/selfmodel.py`：Self-Model（预测 `next_obs / reward / energy / cause`，生成第三因子）。
- `meta/autoadapt.py`：元控制（plateau 检测、UCB 选改动、A/B 评测、回滚、代码热补丁、增删神经元、inner_steps 调整）。
- `scripts/train_xor.py`：XOR 任务（最小可行）。
- `scripts/train_gridworld.py`：探索 + 在线 e-prop。
- `scripts/train_smsa.py`：Self-Model + 自改（“像自我”的闭环）。
- `scripts/snn_text_lm.py`：字符级外部语料（NTP，自监督，可选）。
- `tools/logger.py`：标准库日志 / CSV。

---

## 快速开始

**环境要求**：Python 3.10+，纯标准库，无需额外安装。

### 1) XOR（最小可行 SNN）

```bash
python scripts/train_xor.py
# 期望：20~30 个 epoch 后，XOR 准确率 ≥ 0.9
```

### 2) GridWorld（主动探索 + 在线 e-prop）

```bash
python scripts/train_gridworld.py --episodes 80 --seed 0 \
  --eta_e 0.01 --lam_e 0.95 --inner_steps 10 --intrinsic_beta 0.1
# 期望：平均回报随回合上升，终点成功率 ≥ 60%；当停滞时出现 [meta] 自改 / 回滚日志
```

### 3) SMSA（Self-Model + 自调参 / 热补丁）

```bash
python scripts/train_smsa.py --episodes 80 --seed 0
# 期望：Self-Model 的 next_obs NLL 下降、cause_acc ≥ 0.7；
#       出现 [meta] 自改（如 v_th / eta / inner / patch surrogate 等），Δ < 0 则回滚
```

### 4) （可选）字符级外部语料：语言建模（NTP）

```bash
# 准备 UTF-8 文本或 .gz 至 data/ 目录
python scripts/snn_text_lm.py --corpus_glob "data/*.txt" \
  --vocab_size 512 --n_in 256 --n_hidden 64 --inner_steps 2
# 期望：loss / token 与 ppl 稳定下降
```

**训练脚本常用参数**

- `--episodes`：训练回合数（交互式任务）。
- `--seed`：随机种子（环境 / 网络复现）。
- `--eta_e`, `--lam_e`：e-prop 学习率与资格迹衰减。
- `--inner_steps`：每个观测的内部积分步数（积分更稳）。
- `--intrinsic_beta`：新奇奖励强度。
- `--corpus_glob`：语料通配（文本 / `.gz`）。

---

## 指标与日志

- **任务表现**：平均回报、成功率、到达终点时间。
- **学习情况**：XOR 准确率、语言建模的 loss / token、ppl。
- **内省与自我**：Self-Model 的 next_obs NLL、`cause_acc`（自因 / 他因）。
- **能耗评估**：尖峰总数、平均发放率（代理能耗近似）。
- **自改质量**：触发频率、Δ 回报 / Δ ppl 分布、回滚比例。
- **日志输出**：默认打印到控制台，可写 CSV 至 `runs/` 目录；`runs/exp_*/metrics.csv` 包含回合、回报、能耗、NLL、`meta_action`、`delta`、`reverted` 等列。

---

## 测试

```bash
python -m pytest -q
# 覆盖：LIF 触发 / 复位、资格迹更新、策略 REINFORCE 收敛、元控制 UCB / A-B / 回滚
```

---

## 安全与合规

- 回滚优先：所有自改先 A/B 小样本评测，通过才保留。
- 能耗与稳定：评分中惩罚高发放率与不稳定。
- 外部语料：仅使用有权限的数据，日志不导出原文片段。

---

## 路线图（优先级）

- **P0**：GridWorld 环境、策略头、元控制最小闭环、在线 e-prop、基本单测。
- **P1**：Self-Model、合成学习信号、代码热补丁 / 增删神经元 / inner_steps 的自改与回滚。
- **P2**：外部语料（字符级 NTP）、域采样（UCB）+ 自调参、回放 / 梦想（可选）。
