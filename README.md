# Agentic-OPD

**Recovering Lost Agent Capabilities by On-Policy Distillation for Long-Horizon Tasks**

强化学习已成为训练 long-horizon LLM agent 的主流范式，但一次训练的价值并不止于
最终的最优策略。在 LLM agent 的 GRPO 训练中，验证集上表现最好的 checkpoint 会在
同一次训练中更早的 checkpoint 已经解决的问题上失败；而所有 checkpoint 覆盖的问题
并集，远多于其中任何一个。只部署单个 checkpoint，等于丢弃了这次训练已经获得的能力。

Agentic-OPD 在强化学习训练结束之后回收这些丢失的能力：以验证集最优 checkpoint 作为
学生，其余 checkpoint 冻结后构成教师候选池，通过两个互补模块把散落的能力蒸馏回学生。

- **HCRN（Highest Coverage-Ratio Next）** — 在问题级动态调度教师，选出对学生能力
  补充最大的 checkpoint，并用加权机制避免教师饥饿。
- **LAD（Locality-Aware Distillation）** — 在 rollout 级做动态路由，只在教师确实更强
  的地方填补学生的能力缺口。

在 6 个 long-horizon agent 领域、27 个 benchmark 上，Agentic-OPD 的平均表现持续超过
用于初始化它的最优 checkpoint，最终性能达到甚至超过头部商业闭源模型。代码与模型权重
将会开源。
