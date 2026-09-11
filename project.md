# Search-OPD

## 项目定位

Search-OPD 用同一 Search-R1 训练 run 的时间序列 checkpoint 作为冻结教师，
训练一个可部署的 Qwen2.5-3B-Instruct 学生模型。研究目标是观察 Search-R1
训练中的题目级搜索能力是否发生遗忘，并验证 temporal OPD 能否把不同阶段的
能力合并到一个学生模型中。

实验专属代码位于 `code/search_opd/`，底层复用 `code/verl-tool` 的
Search-R1 数据、Ray、FSDP2、vLLM 异步 rollout、Search tool 和 reward
manager。除非确有必要，不要修改原有 RL 入口或改变其行为。

## 当前实验协议

### 模型与教师

- 学生：`Qwen2.5-3B-Instruct`，默认路径
  `/ssd2/llm_models/Qwen2.5-3B-Instruct`。
- 教师：同一 Search-R1 run 的不可变 HuggingFace checkpoint；当前验证 run 使用
  `global_step_5` 至 `global_step_500` 的 100 个时间点。
- 完整教师清单、顺序和来源保存在 manifest；GPU reference worker 只驻留一个
  冻结教师。每次 validation 后可原子切换 resident teacher。
- 启动 worker 前会按 validation capability set 删除被其他教师完全覆盖的
  checkpoint；相等集合也视为被覆盖，但 active teacher list 始终非空。

### 数据角色

- 训练：`data/search_r1/training_data/train.parquet`。
- Validation 与自适应选师：`data/search_r1/test/nq.parquet`、
  `data/search_r1/test/hotpotqa.parquet`。
- Held-out test 与 checkpoint retention：`bamboogle.parquet`、
  `musique.parquet`、`triviaqa.parquet`。
- NQ/HotpotQA 不得被当作 teacher 之外的 test 选择依据；三个 held-out
  数据集不参与 teacher selection。

每 5 个 training steps 做一次 validation、test 和 teacher selection，最多训练
300 steps。validation 使用 deterministic `n=1` rollout；训练 rollout 使用
`n=5`。

### 自适应教师选择

令学生当前 solved question set 为 `P`，教师 `i` 的 solved set 为 `Qi`：

- 每道 validation question 初始权重为 1；
- 每轮对所有 active teacher 的 `Qi - P` 取并集，集合中的每道题只加 1；
- 权重上限为 5；
- 教师分数为
  `sum(weight[q] for q in Qi - P)`，取分数最高者；
- 权重、完整 ranking、active/pruned 清单和历史写入
  `adaptive_teacher_selection.json`；
- state 文件通过原子替换发布，避免 reference worker 读到半成品。

### OPD 与运行时

当前目标是 sampled-token PG-OPD，而不是 full-vocabulary KL。教师在学生生成的
相同 prefix 上计算 sampled token log-probability；tool observation 保留在
prefix 中，但不进入 generated-token OPD mask。默认 task loss 为 0，更新只由
OPD 提供。

默认关键配置：

- 8 张 NVIDIA A800 80 GB；`verlai/verl:vllm011.latest` Docker；
- FSDP2、AdamW、learning rate `1e-6`、warmup ratio `0.285`、weight decay
  `0.01`；
- train batch 与 PPO mini-batch 均为 `512`，每 GPU actor micro-batch 为
  `16`，`ppo_epochs=1`，关闭 dynamic batch；
- prompt length `4096`，每轮 response `500`，最多 4 个 assistant turns，
  generated response 上限 `2000`，observation 上限 `500`，trajectory 上限
  `4096`；
- asynchronous vLLM `tool_agent`，Search-R1 multi-turn format，temperature
  `1.0`，top-p `1.0`；
- `OPD_LOSS_COEF=1.0`、`OPD_TASK_LOSS_COEF=0.0`、
  `OPD_MAX_ABS_LOG_RATIO=null`，不使用 critic 或 KL reward。

Retriever 默认使用 `http://127.0.0.1:8181/retrieve`。启动脚本会先执行
teacher/student readiness preflight 和 retriever preflight；服务配置来自
`/ssd2/chengmingquan/Search-R1/retrieval_launch.sh`，索引、语料和 embedding
模型路径见该脚本及 `code/search_opd/README.md`。

## 启动方式

必须提供且只能提供一种教师来源：

```bash
OPD_TEACHER_MANIFEST=/path/to/teacher_manifest.json \
MODEL_PATH=/ssd2/llm_models/Qwen2.5-3B-Instruct \
bash code/search_opd/train.sh
```

或：

```bash
OPD_TEACHER_MODEL_PATHS=/path/to/global_step_5,/path/to/global_step_10 \
bash code/search_opd/train.sh
```

设置 `TEACHER_CHECKPOINT_ROOT` 时，脚本可自动按数值顺序发现其中的
`global_step_*` 目录。manifest 应记录每个教师的 `name`、`path`、`step`，
并在可用时记录 SHA-256。

## Checkpoint retention

学生 checkpoint 严格按三个 test set 的等权平均 EM 选择：

```text
(Bamboogle EM + Musique EM + TriviaQA EM) / 3
```

只有 validation-aligned 的周期 step 才允许保存。候选先写入 staging 目录，
验证完整后原子发布；分数更高或同分但 step 更新的候选会替换旧结果。输出目录
最多保留一个 `global_step_*` HuggingFace checkpoint。

主要审计文件：

- `opd_run_manifest.json`：学生、教师、数据、seed、rollout、目标函数和训练
  配置；
- `adaptive_teacher_selection.json`：加权选师 state；
- `checkpoint_selection.json`：retention 策略、候选和最终结果；
- `validation_checkpoint_metrics.jsonl`：每个候选的三项 EM；
- `validation_trajectories/`、`test_trajectories/`：各 step 的 trajectory。

## 关键代码

- `train.sh`：唯一启动入口、环境变量和 preflight；
- `main_ppo_opd.py`：独立 Ray/Hydra 入口；
- `core_opd.py`、`dp_actor_opd.py`：sampled-token OPD、mask 和逐教师 loss
  聚合；
- `fsdp_workers_opd.py`：单 resident 冻结教师及动态切换；
- `adaptive_teacher_opd.py`：capability pruning、加权边际覆盖和原子 state；
- `checkpoint_trainer_opd.py`、`checkpoint_retention_opd.py`：validation、
  test、teacher 发布和单 checkpoint retention；
- `config_opd.py`、`preflight_opd.py`、`teacher_readiness_opd.py`：manifest、
  配置约束和启动前检查；
- `test_*.py`：选师、数据协议、mask、actor protocol 和 retention 的 focused
  tests。

详细的目标函数、启动检查和测试说明见 `code/search_opd/README.md`。

## A800 验证里程碑

2026-08-24 在 A800 上运行：

`/ssd1/tcbian/Search-OPD/opd_weighted_qwen25_3b_20260824_retry1`

已核验到 `global_step=20`：

- validation/selection history 为 `[0, 5, 10, 15, 20]`；
- step 0 发生首次切换：
  `teacher_00_global_step_5 -> teacher_99_global_step_500`；
- step 20 仍选择 `teacher_99_global_step_500`，active teachers 为 100，
  pruned teachers 为 0，最大 question weight 为 5；
- OPD actor 在 step 1、5、10、15、20 等更新中均为单 teacher
  (`teacher_count=1`)；
- step 20 的 test EM：
  `Bamboogle=0.208000`、`Musique=0.063302`、
  `TriviaQA=0.513922`，等权均值 `0.261741`；
- step 5、10、15、20 的 retention 均值分别为
  `0.251560`、`0.243933`、`0.245590`、`0.261741`，因此最终仅保留
  `global_step_20`；
- 运行日志中未发现 `Traceback`、`RuntimeError`、`ValueError` 或 teacher
  tensor/config mismatch。

这是已完成 step 20 的工程验证里程碑，不代表 300 steps 或多 seed 的最终研究
结论。

## 解释边界与后续

当前协议复用了 NQ/HotpotQA 的 test parquet 做 validation 和 solved-UID
选师，因此属于 test-informed diagnostic，不能作为干净 held-out
generalization 结果。Bamboogle、Musique、TriviaQA 虽不参与选师，但参与
checkpoint retention，也不能被描述为完全 untouched 的最终测试。

旧的 SearchOPDMuon RL run 是独立 baseline，不得与本 OPD run 混称。后续若要
形成研究结论，应补充固定 train/validation/test manifest、数据 overlap audit、
至少多个 seed、uniform/random temporal controls、matched student/end-to-end
compute，以及不使用 test 数据做模型选择的 clean protocol。模型 checkpoint
和生成 corpus 不提交仓库，只保留不可变路径与运行 manifest。

## 论文写作进展与决策记录（2026-09-07/08）

投稿目标：ICLR 2027（摘要截止 2026-09-11，全文截止 2026-09-16，正文 9 页），
论文源码在 `paper/Agentic-OPD/`，格式要求见 `paper/Agentic-OPD/instructions.md`。

### 已完成

- 标题已从 "Scaling Agent Capabilities via On-Policy Distillation from
  Student-Routed Temporal Teachers" 改为 **"Agentic-OPD: Recovering
  Forgotten Agent Capabilities by Distilling from Historical Checkpoints"**
  （`Agentic-OPD.tex`）。理由：论文核心是 consolidation/recovery 而非
  scaling；"Student-Routed Temporal Teachers" 拗口。
- `abstract.tex` 已重写。关键口径：
  - 定位为 **RL 之后的融合阶段**（"a consolidation stage that follows
    reinforcement learning without modifying it"），这是用户确定的卖点；
  - 摘要**不写具体数字**（用户决定），只声称超过 pure-RL（GRPO）；投稿前
    数字锁定后可以再加一句 "by up to X points"；
  - 已删除旧摘要中 "$X$ 占位符"、"commercial models" 对比承诺和
    "practical path" 等空话；
  - 注意："outperforms ... across seven QA benchmarks" 隐含七个集全赢；
    若最终 2Wiki/MusiQue 仍略输 GRPO，需弱化为 "on average"。

### 实验数据来源与可信度（重要）

- 实验数字在内部知识库文档：
  `https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/_SKPgSwp2G/-7bgqLe28I/Z5qlcE2GTxF53x`
  （doc-id `Z5qlcE2GTxF53x`，用 ku-doc-manage skill 的
  `ku query-content --doc-id Z5qlcE2GTxF53x` 读取；WebFetch 打不开）。
- **可信数字**：DeepSearch 主对比（Qwen2.5-3B-it 组，7 个 QA 集 EM）：
  ours 平均 38.4 vs Search-R1/GRPO 36.2（+2.3），7 集赢 5，2Wiki 和
  MusiQue 略输。
- **作废数字**：文档中的消融行（fixed-teacher best/merge、random teacher、
  adaptive unweighted，其中 unweighted 38.9 > ours 38.4）和 ReTool 表
  （ours 低于 GRPO）是实习生用有 bug 的 AI 生成代码跑的，**逻辑错误、
  不可引用**。用户手上有真实结果：方法有效、加权消融符合预期。真实数字
  尚未提供（截至 09-08）。知识库文档保持原样，不去修正。

### 九页实验布局（已与用户确认）

- 主实验：DeepSearch 详细（表 1：NQ/HotpotQA/PopQA/TriviaQA/2Wiki/
  MusiQue/Bamboogle，行含 base、GRPO、fixed-teacher latest/best、
  merge（GTR-Turbo 式）、random routing、adaptive unweighted、ours）。
- 其余五领域（code generation、embodied AI、WebShop、tool use=ReTool、
  data analysis）只比 **base / GRPO / ours** 三行（表 2，紧凑跨领域表）。
- 闭源/大模型行和 7B 组放 appendix 或删除；主文只在 3B 组内比 EM
  （闭源模型 EM 被答案格式压低，用 EM 比不公平，而 ours 无 subEM 数据）。
- ReTool 领域用 AIME 2024/2025，只有 30 题，必须报 avg@16 或 avg@32
  加方差，且 base/GRPO/ours 三行评测协议对齐。
- DeepSearch 表中 Qwen2.5-3B-it base 行目前为空，是三方对比锚点，必须补。

### 方法讨论结论（2026-09-07）

- 现行设计：题目级 "学生错、教师对" 判断只发生在**教师选择层**
  （验证/routing 集上）；训练时教师不在训练集上做 rollout，只在学生
  生成的 prefix 上算 logits；OPD loss 施加于所有训练 rollout。
- 新想法（用户提出）：**wrong-only OPD**——用训练 rollout 已有的 result
  reward 做 rollout 级 mask（零额外成本），只在学生做错的 rollout 上蒸馏。
  决定：先做消融（all rollouts vs failed-only，可加第三档：对的 rollout
  给 0.1 小系数），若显著更优再升级为主方法。实现点：`core_opd.py` 的
  generated-token mask 乘 per-rollout 的 1[r=0]。
  风险记录：错误 prefix 上教师信号质量无保证；学生变强后有效 batch 萎缩
  （参考 DAPO dynamic sampling）。若升级为主方法，`method.tex` 结尾
  "retain verified-correct student trajectories for replay" 一句需同步修改。

### 待办

1. 等用户提供真实消融数字和真实 ReTool 表，然后搭 `experience.tex`
   （目前该文件和 `conclusion.tex` 均为空）。
2. intro 叙事统一为 "post-RL consolidation stage" 口径（目前只有摘要改了）；
   intro 贡献第 4 条 "matched student-update and end-to-end compute budgets"
   声明需核实实验是否真的做到，做不到则删。
3. method.tex 声称 D_route 与 train/test disjoint，但实际实现用 NQ/HotpotQA
   test parquet 做 routing（见上文"解释边界"节），投稿前必须洗成 clean
   protocol，否则论文声称与代码不一致。
4. preliminary 六宫格（`preliminary.tex`）目前只有 DeepSearch 一格有图，
   其余五格是 placeholder；画 gap 曲线要求每个领域都有存密集 checkpoint 的
   GRPO run，可行性待确认，不行则收缩格数。
5. 全文过一遍 AI 味措辞；正文有未使用的 `\taopd` 宏，定稿时清理。
6. ICLR 2027 要求 AI Use Statement（不计页数），投稿前补。
