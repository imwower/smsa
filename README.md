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

#### 能耗惩罚（REINFORCE）实验与复现

- 改动摘要：
  - 在训练环节将优势替换为 `adv ← adv − λ × (spikes/hidden_dim)`；
  - 调度器评分加入 `energy_penalty` 并在 CSV 记录；
  - 训练日志每 10 回合打印 `norm_spikes` 与 `energy_penalty`。

- 快速复现（80 回合）：

  1) 基线（λ=0）

  ```bash
  python scripts/train_gridworld.py --episodes 80 --seed 0 \
    --lambda-energy 0 --homeo off --gamma-energy 0.0
  ```

  2) 加惩罚（命中验收范例）

  ```bash
  python scripts/train_gridworld.py --episodes 80 --seed 0 \
    --lambda-energy 2.0 --homeo off --gamma-energy 0.0 --inner-steps 10
  ```

- 结果摘要（固定 seed=0）：
  - 平均能耗（spikes 均值）：基线 102.41 → 惩罚 79.40（下降 22.47%）
  - 成功率（尾窗）：基线 0.90 → 惩罚 0.90（无下降）
  - 日志示例：`Energy: norm_spikes=0.0184 energy_penalty=0.3672 lambda=2.000`
  - 产物：`runs/gridworld_metrics_baseline_e80_new.csv` 与 `runs/gridworld_metrics_penalty_lambda2_steps10_e80.csv`

> 注：可通过 `--lambda-energy`、`--inner-steps` 微调折中；更长回合（300）下，λ∈[1.2,2.0] 亦能带来 5–9% 的能耗下降且成功率保持 1.00。

#### Post 门控（继续学 + 再试）

- 触发条件：`overall < 0.5` 或 `tokens < 80` 时，守护进程自动进行“继续学 + 再试”。
- 动作细节：
  - 采用微批增量训练：每批 `600` 行，最多累计至 `~3000` 行或超时 `3 分钟`；每训练一批立即再试一次生成。
  - 采样偏向主题域：若给定 `--post-topic`，训练数据优先选用父目录名匹配该主题的文件。
  - 再试时自动注入因果引导种子（含“因为…所以…”句式），提升自解释分与整体分。
- 记录项：
  - `runs/daemon.csv` 新增列：`relearned`（是否继续学）、`retuned`（是否改解码参数）、`patched`（是否补丁）。
  - `runs/self_report.md` 中“校准说明”一行会包含“继续学/改参/补丁/训练”的布尔标记；若空样本（`tokens==0` 或 `spikes==0`），会标注 `[空样本] 解码参数：...`。

复现（直到出现 ≥+0.1 提升的样例，建议 3–5 轮）：

```bash
python scripts/daemon.py --loops post --poll-seconds 0 \
  --max-iterations 5 --post-len 60 --post-topic 学习 \
  --post-threshold 0.5 --post-min-self 0.4
```

期望：
- `runs/daemon.csv` 出现一行包含 `retrain_then_retry=True`，并在 `note` 中带有 `delta_overall=+0.121`（或 ≥+0.1 的正提升），同时 `relearned=True`。
- `runs/self_report.md` 对应回合的条目显示 `[空样本]`（若为空样本）、“继续学 True；改参 True/False；补丁 True/False；训练 True/False”等话术。

### 3) SMSA（Self-Model + 自调参 / 热补丁）

```bash
python scripts/train_smsa.py --episodes 80 --seed 0
# 期望：Self-Model 的 next_obs NLL 下降、cause_acc ≥ 0.7；
#       出现 [meta] 自改（如 v_th / eta / inner / patch surrogate 等），Δ < 0 则回滚
```

### 4) 外部语料（基于 HF datasets）：语言建模（推荐流程）

1. 安装并展开数据集（默认 suolyer/webqa）

```bash
python scripts/fetch_dataset.py --dataset suolyer/webqa --splits train --output data/hf/webqa
```

2. 将语料路径写入统一配置并训练（推荐方案 1）

```bash
# LM（守护进程可读取配置）：
python scripts/daemon.py --loops lm --poll-seconds 0 --lm-lines 100 \
  --max-iterations 3 --corpus-path data/hf/webqa/train.txt

# GridWorld / SMSA 训练也可在启动时注入 --corpus-path（自动写入配置）：
python scripts/train_gridworld.py --episodes 80 --corpus-path data/hf/webqa/train.txt
python scripts/train_smsa.py --episodes 80 --seed 0 --corpus-path data/hf/webqa/train.txt
```

3. 统一配置位置

- 路径：`runs/datasets_config.json`
- 脚本在启动时读取该文件并注册语料（`inputs` 或 `output_dir/train.txt`）

**训练脚本常用参数**

- `--episodes`：训练回合数（交互式任务）。
- `--seed`：随机种子（环境 / 网络复现）。
- `--eta_e`, `--lam_e`：e-prop 学习率与资格迹衰减。
- `--inner_steps`：每个观测的内部积分步数（积分更稳）。
- `--intrinsic_beta`：新奇奖励强度。
- `--corpus-path`：单文件路径；脚本会写入统一配置，训练时按配置加载语料。

---

## 指标与日志

- **任务表现**：平均回报、成功率、到达终点时间。
- **学习情况**：XOR 准确率、语言建模的 loss / token、ppl。
- **内省与自我**：Self-Model 的 next_obs NLL、`cause_acc`（自因 / 他因）。
- **能耗评估**：尖峰总数、平均发放率（代理能耗近似）。
- **自改质量**：触发频率、Δ 回报 / Δ ppl 分布、回滚比例。
- **日志输出**：默认打印到控制台，可写 CSV 至 `runs/` 目录；`runs/exp_*/metrics.csv` 包含回合、回报、能耗、NLL、`meta_action`、`delta`、`reverted` 等列。

- 报告话术规则（tools/reporter.py）：
  - 置信度文案映射：`conf < 0.25` → “偏低”；`0.25 ≤ conf ≤ 0.75` → “基本匹配”；`conf > 0.75` → “偏高”。
  - 空样本标注：当 tokens==0 或 spikes==0 时，在 `runs/self_report.md` 标注“[空样本] 解码参数：T=… top_k=… repeat_penalty=… trigram=…”。
  - RL 能耗箭头：核心指标后追加“（能耗 vs 上一回合：↑/↓/→）”，基于 `runs/daemon.csv` 最近两条 RL 记录的 `spikes` 对比。
  - 校准话术示例：
    “我对下一观测的置信度为 {conf:.2f}；过去 50 回合的校准相关 ρ={rho:.2f}，说明我的置信{偏高/偏低/基本匹配}。”

---

## 测试

```bash
python -m pytest -q
# 覆盖：LIF 触发 / 复位、资格迹更新、策略 REINFORCE 收敛、元控制 UCB / A-B / 回滚
```

### 快速回归（CI 建议顺序）

1. `python scripts/smoke_env.py` — 运行最小随机策略冒烟，快速确认 GridWorld 环境输出稳定。
2. `python scripts/train_gridworld.py --episodes 30` — 观察平均回报是否上升，并确认 `[meta]` 自改触发记录。
3. `python scripts/train_smsa.py --episodes 30` — 确认 Self-Model 的 `nll` 与 `cause_acc` 指标走势正常（NLL 下降、cause_acc ≥ 0.7 目标）。
4. `python -m pytest -q` — 单元测试全绿。

> 以上步骤可直接写入 CI，确保脚本 + 单测在变更后始终可运行。

---

## 安全与合规

- 回滚优先：所有自改先 A/B 小样本评测，通过才保留。
- 能耗与稳定：评分中惩罚高发放率与不稳定。
- 外部语料：仅使用有权限的数据，日志不导出原文片段。

---

## 监督 + 持续学习 + 自改 + 自发输出（本地 CPU 长跑）

以下命令适合在本地 CPU 上进行“监督 + 持续学习 + 自改 + 自发输出”的联合长跑，所有产物均落地到 `runs/` 目录，便于审计与复盘。

1) 准备语料（两种方式）

- 推荐：使用脚本拉取并展开数据集，然后通过 `--corpus-path` 写入统一配置，守护进程会读取该配置。

```bash
# 例：拉取 suolyer/webqa 并展开到 data/hf/webqa
python scripts/fetch_dataset.py --dataset suolyer/webqa --splits train --output data/hf/webqa
# 通过启动参数写入 runs/datasets_config.json（之后可不再传 --corpus-path）
python scripts/daemon.py --loops lm --poll-seconds 0 --lm-lines 50 \
  --max-iterations 1 --corpus-path data/hf/webqa/train.txt
```

或使用内置随机可读语料（模板+同义词扰动）

```bash
# 生成 3 份可读中文种子语料：data/seed_*.txt（每份 ≥800 行）
python - <<'PY'
from tools.corpus_seed import write_corpus, random_topics
for i,t in enumerate(random_topics()[:3]):
    write_corpus(f"data/seed_{i}_{t}.txt", lines=800, topic_hint=t)
print("seed corpora ready.")
PY
```

2) 启动守护进程（RL + LM + 自改 + 发帖 + 解释性门控）

```bash
python scripts/daemon.py --loops rl,lm,autoadapt,post --poll-seconds 20 \
  --rl-episodes 30 --lm-lines 1200 --post-len 200
```

观察产物（可审计）

- runs/daemon.csv：联合循环的逐轮指标
- runs/scheduler.csv：调度器奖励/能耗与下一步预算
- runs/lm.csv：语言建模训练曲线（loss、ppl、Δppl、spikes）
- runs/calibration.csv：自我模型校准指标（若相关脚本记录）
- runs/autopatch.log：自动补丁引擎日志（锚点内修改 + A/B + 回滚）
- runs/feed/*.md：自发“脉冲式”文本内容（每篇末尾包含 Explainability 指标）
- runs/self_report.md：每轮中文说明（做了什么、为何做、效果如何、是否回滚、下一步计划）
- 解释性不足时：自动调参/训练/补丁 → 重试 → 记录 runs/explain_log.md（“观测→诊断→干预→结果→下一步”）

风控与否决（QA/Contracts）

- `tools/contracts.py` 实施补丁类别白名单（surrogate / defaults / candidates / decode）与锚点外改动禁止；参数边界：eta_e∈[1e‑4,0.1]、lam_e∈[0.5,0.999]、inner_steps∈[4,30]、top_k∈[10,128]、repeat_penalty∈[1.0,2.0]
- 强制阈值（满足其一，否则回滚）：
  - post：explainability overall ≥ +0.05 且 |Δ能耗| ≤ 10%
  - lm：perplexity 相对下降 ≥ 1.5%
  - rl：平均回报提升 ≥ +0.02
- 否决即回滚：任一条件失败将自动 revert() 并在 `runs/self_report.md` 记录否决原因

---

### 后台运行与有界长跑示例

- 有界长跑（示例 50 轮）：

```bash
python scripts/daemon.py --loops rl,lm,autoadapt,post \
  --poll-seconds 10 --rl-episodes 20 --lm-lines 800 --post-len 180 \
  --max-iterations 50
```

- 使用 screen（跨平台简单）：

```bash
screen -S smsa-daemon
python scripts/daemon.py --loops rl,lm,autoadapt,post --poll-seconds 20 \
  --rl-episodes 30 --lm-lines 1200 --post-len 200
# 按下 Ctrl-A 然后 D 以分离；恢复：screen -r smsa-daemon
```

- 使用 tmux（推荐）：

```bash
tmux new -s smsa
python scripts/daemon.py --loops rl,lm,autoadapt,post --poll-seconds 20 \
  --rl-episodes 30 --lm-lines 1200 --post-len 200
# 分离：Ctrl-B 然后 D；恢复：tmux attach -t smsa
```

- 使用 systemd（Linux 用户，用户级服务）

`~/.config/systemd/user/smsa-daemon.service`：

```ini
[Unit]
Description=SMSA Background Daemon

[Service]
WorkingDirectory=%h/code/github/smsa
ExecStart=/usr/bin/env python scripts/daemon.py --loops rl,lm,autoadapt,post \
  --poll-seconds 20 --rl-episodes 30 --lm-lines 1200 --post-len 200
Restart=always

[Install]
WantedBy=default.target
```

启用并启动：

```bash
systemctl --user daemon-reload
systemctl --user enable --now smsa-daemon
journalctl --user -u smsa-daemon -f  # 查看日志
```

## 路线图（优先级）

- **P0**：GridWorld 环境、策略头、元控制最小闭环、在线 e-prop、基本单测。
- **P1**：Self-Model、合成学习信号、代码热补丁 / 增删神经元 / inner_steps 的自改与回滚。
- **P2**：外部语料（字符级 NTP）、域采样（UCB）+ 自调参、回放 / 梦想（可选）。
### Post 联动策略（连续不达标时的自动干预）

- 触发条件：连续 N 次（默认 2，`--post-fail-max` 可调）post 总分 `overall` 未达阈值（`--post-threshold`）或 tokens<80。
- 联动动作（按顺序）：
  1) 强制执行 `code_patch:decode_trigram_on`（A/B 验证后保留或回滚，详见 runs/autopatch.log 与 runs/contracts_log.csv）。
  2) 继续学 `train_lines(L)`，默认 `L=1500`，可用 `--link-lm-lines` 指定；域选择优先匹配 `--post-topic`（文件名/元数据）。
  3) 立即再生成一次 post（记录 Δoverall）。
- 审计与报告：
  - `runs/scheduler.csv` 追加一条 `task=link` 的“联动”事件，`note` 包含 “联动: trigram_on + lm{L} + post_retry Δoverall=...”。
  - `runs/self_report.md` 写入本次 post 再试条目；Reporter 会输出校准话术、空样本标注与（若为 RL）能耗箭头。

示例（更易触发联动）：

```bash
python scripts/daemon.py --loops post --poll-seconds 0 --max-iterations 12 \
  --post-len 160 --post-topic 学习 --post-fail-max 2 --post-threshold 0.80 \
  --link-lm-lines 1500
```
