# SMSA — Self-Modeling Spiking Agent (标准库版)

> 只用 Python 标准库 + 普通 CPU 的 **脉冲神经网络 (SNN)** 智能体：主动探索、在线学习 (e-prop 三因子)、自我模型 (Self-Model)、自调参与微演化、代码热补丁（替代导数）。

## 目录结构
```text
├─ README.md
├─ envs/
│ └─ gridworld.py # 5×5 GridWorld，标准库实现
├─ snn/
│ ├─ lif.py # LIF/ALIF 神经元与替代导数
│ ├─ dense.py # 稠密层 + e-prop eligibility
│ ├─ policy.py # 策略头 (REINFORCE)
│ └─ selfmodel.py # Self-Model 网络与读出头
├─ meta/
│ └─ autoadapt.py # UCB 多臂赌博机、自调参、结构微演化、A/B 回滚、代码热补丁
├─ scripts/
│ ├─ train_xor.py # 阶段A：XOR 训练
│ ├─ train_gridworld.py # 阶段B：主动探索 + 在线学习
│ ├─ train_smsa.py # 阶段C/D：Self-Model + 自调参/热补丁
│ └─ eval_metrics.py # 统一指标评测与报表
├─ tools/
│ └─ logger.py # 纯标准库 CSV 日志、简单可视化（ASCII）
└─ tests/
   ├─ test_lif.py
   ├─ test_eprop.py
   └─ test_meta.py
```

也可以直接使用单文件原型：
> - `snn_py_evo.py` —— 阶段A（XOR）  
> - `snn_py_explore_auto.py` —— 阶段B/D（探索 + 自调参/热补丁）  
> - `snn_self_agent.py` —— 阶段C/D（Self-Model + 自改 + 回滚）  
> 三者均仅依赖标准库，`python 文件名.py` 即可运行。

## 快速开始

> 无需安装第三方库；建议使用 Python 3.10+。

```bash
# 1) XOR（阶段A）
python snn_py_evo.py
# 预期：20~30 个 epoch 后，XOR acc ≥ 0.9

# 2) 主动探索 + 自改（阶段B/D）
python snn_py_explore_auto.py
# 预期：控制台每 10 回合打印平均回报；停滞时输出 [meta] 自改与回滚日志

# 3) Self-Model + 自改（阶段C/D）
python snn_self_agent.py
# 预期：Self-Model 的观测预测 NLL 下降；自因/他因分类准确率升高；[meta] 日志显示自改闭环
```

## 关键概念

> - e-prop 三因子：局部 eligibility × 第三因子（来自策略/自我模型的误差信号），实现在线学习。
> - 内在动机：基于访问计数的新奇奖励，鼓励主动探索。
> - Self-Model：预测下一观测、奖励、能耗，以及“性能变化来自自改还是环境”的二分类。
> - 自调参与微演化：学习率、阈值、内在奖励权重、inner-steps、增删神经元、替代导数切换/热补丁；A/B 小样本评测，不佳即回滚。
> - 能耗代理：使用尖峰总数/发放率作为能耗近似指示。

## 命令行参数（脚本示例）

`train_gridworld.py`（若采用分模块结构）：

```bash
python scripts/train_gridworld.py \
  --episodes 80 --seed 0 \
  --intrinsic_beta 0.1 --eta_e 0.01 --lam_e 0.95 \
  --inner_steps 10 --poisson_on 0.2 --poisson_off 0.02 \
  --stochastic_slip 0.1 --goal_reward 1.0
```

## 配置（仅标准库）

- 环境：`envs/gridworld.py` 中可调 `w`, `h`, `goal`, `max_steps`, `stochastic_slip`, `step_penalty`, `goal_reward`。
- 网络：`lif.py` / `dense.py` 中可调 `v_th`, `tau_m`, `tau_a`, `pseudo_width`, `pseudo_gain`, `tau_syn`, `eta_e`, `lam_e`。
- 策略：`policy.py` 中调整学习率 `lr`。
- 自我模型：`selfmodel.py` 中设置读出头权重、损失加权。
- 元控制：`meta/autoadapt.py` 中配置 UCB 候选集合、plateau 窗口、A/B 回合数、回滚策略。

## 指标与期望

- 任务表现：平均回报、成功率、时间到目标。
- 能耗：每回合尖峰总数、平均发放率。
- 内省：Self-Model 的观测预测 NLL、置信度与准确率相关性。
- 自因归因：自因/他因分类准确率。
- 自改质量：触发自改频率、Δ回报分布、回滚比例。
- 日志：`runs/exp_*/metrics.csv`，包含回合号、回报、能耗、NLL、`cause_acc`、`meta_action`、`delta`、`reverted` 等列。

## 可视化（标准库）

`tools/logger.py` 用 CSV 写入，并以 ASCII 简图（均值/移动平均）在控制台打印趋势。若需图片，可引入 matplotlib，但默认不依赖第三方库。

## 故障排查

- 不发/全发：调 `v_th`、`poisson_on/off`、`tau_m/tau_a`；监控发放率并自适应升阈值。
- 训练停滞：增大 `inner_steps`、略升 `eta_e`、提高 `intrinsic_beta`；确认 plateau/AB 参数不过于保守。
- 自改震荡：拉长 plateau 窗口、增加 A/B 回合数、为同类改动设置最小间隔（cooldown）。

## 安全与可审计

- 改动应先评测再采纳，并保持可回滚。
- 日志需包含改动类型、触发原因、Δ回报、是否保留/回滚。
- 将能耗与稳定性纳入评分，避免“为性能牺牲稳定与能耗”。

## 许可与声明

本仓库为研究/教学用途；不保证在所有场景稳定。请在可控环境下测试和部署。
