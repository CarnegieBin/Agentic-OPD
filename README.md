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

## 目录结构

```
Agentic-OPD/
├── search_opd/     Search-R1 域上的多教师 OPD 实现，唯一启动入口 train.sh
└── verl-tool/      底层 RL 框架（verl / verl_tool）
```

`search_opd/` 是独立的 Search-R1 多教师 OPD 入口，不修改 `verl-tool` 中的已有代码，
只包装现有的 dataset、Search agent、奖励函数、验证与 checkpoint 逻辑。当前实现为
sampled-token PG-OPD：在学生 rollout 上采样 reverse-KL，将其负值作为 stop-gradient
的 token advantage，多教师先分别求损失再等权平均。教师调度按题目级边际覆盖进行，
每次学生完成验证后重新发布下一个教师。

`verl-tool/` 提供 dataset、异步 tool rollout、FSDP worker 与 reward manager；
底层 actor、FSDP、异步 rollout 和验证代码均复用该目录。

## 启动

教师必须是同一次 Search-R1 训练留下的不可变 HuggingFace checkpoint，顺序即聚合顺序：

```bash
OPD_TEACHER_MANIFEST=/path/to/teacher_manifest.json \
MODEL_PATH=/ssd2/llm_models/Qwen2.5-3B-Instruct \
bash search_opd/train.sh
```

启动前会依次执行 teacher/student readiness preflight 与 retriever preflight，拒绝
不完整的 checkpoint、未完成的 merge 以及与训练输出目录重叠的教师路径。完整的环境变量、
checkpoint 选择规则（Bamboogle / Musique / TriviaQA 三测试集等权平均 EM）和自适应
教师调度算法见 [`search_opd/README.md`](search_opd/README.md)。

## 测试

`search_opd/test_*.py` 是单元测试，需要在装有 verl-tool 训练依赖的环境中运行：

```bash
cd search_opd && python -m pytest test_*.py
```

## 论文

论文源码、图表与 preliminary 数据位于同级目录 `../paper-Agentic-OPD/`，
未包含在本仓库中。编译：

```bash
cd ../paper-Agentic-OPD && latexmk -pdf -interaction=nonstopmode Agentic-OPD.tex
```
