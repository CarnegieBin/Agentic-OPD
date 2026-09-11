# Search-OPD 多教师训练

这个目录是独立的 Search-R1 多教师 OPD 入口。`code/verl-tool` 中已有
代码不需要修改；本目录只包装现有的 dataset、Search agent、奖励函数、验证
和 checkpoint 逻辑。

## 目标函数

给定学生在 prompt 上生成的 token $y_t$，教师 $i$ 在完全相同的学生前缀
$s_t$ 上计算该 token 的 log-probability：

$$
\widehat{D}_{i,t}
  = \log \pi_S(y_t\mid s_t)
    - \log \pi_{T_i}(y_t\mid s_t).
$$

这是在学生 rollout 上采样的 reverse-KL estimator。PG-OPD 将其负值作为
stop-gradient token advantage：

$$
A^{\mathrm{OPD}}_{i,t}
  = \operatorname{sg}\left[
      \log \pi_{T_i}(y_t\mid s_t)
      -\log \pi_S(y_t\mid s_t)
    \right].
$$

每个时刻只在 reference worker 中驻留一个教师；学生每次在 NQ 和
HotpotQA 验证后，driver 根据题目级边际覆盖重新发布下一个教师。默认
`OPD_TASK_LOSS_COEF=0`，因此训练更新只来自 OPD；QA exact-match reward
仍由原 Search-R1 reward manager 计算，用于日志和验证。需要联合 GRPO 时可
将 `OPD_TASK_LOSS_COEF` 设为正数。

为使多教师“先分别求损失、再平均”保持单次 on-policy 更新，入口约束
`ppo_epochs=1`、`ppo_mini_batch_size=train_batch_size`、关闭 dynamic batch；
这也是当前脚本的配置。底层 actor、FSDP、异步 tool rollout 和验证代码均
来自现有 `verl-tool`。

多轮 Search-R1 trajectory 的 observation 仍保留在每个后续 assistant
generation 的 `input_ids` 前缀中，但 OPD action mask 只允许模型实际生成的
assistant/search/answer token。当前 `ToolAgentLoop` 的 `response_mask` 已按
`1=generated`、`0=tool observation or padding` 生成；actor 会验证其形状、
二值性并额外与 response attention mask 相交。若未来 rollout adapter 同时
传递 `opd_response_mask`，actor 会优先使用该显式 mask。

## 启动

教师必须是同一 Search-R1 run 的不可变 HuggingFace checkpoint，顺序就是
聚合顺序。推荐使用 JSON manifest：

```bash
OPD_TEACHER_MANIFEST=/path/to/teacher_manifest.json \
MODEL_PATH=/ssd2/llm_models/Qwen2.5-3B-Instruct \
bash train.sh
```

也可以用逗号分隔的路径：

```bash
OPD_TEACHER_MODEL_PATHS=/ssd1/run/global_step_30,/ssd1/run/global_step_105 \
bash train.sh
```

脚本中的训练集、验证集、rollout 数量、序列长度和 Search 工具与
`code/verl-tool/scripts/Search-R1/train.sh` 保持一致。NQ/HotpotQA 仅用于
教师调度和 validation；Bamboogle、Musique、TriviaQA 是独立 test
dataloader，checkpoint 只按三个测试集 EM 的均值保留。

启动前默认会做 retriever preflight；调试时可设置
`SEARCH_R1_SKIP_RETRIEVER_PREFLIGHT=1`。每次运行会在
`CHECKPOINT_DIR/opd_run_manifest.json` 保存教师顺序、student、数据、seed、
rollout 限制、reward/verifier 和 OPD 系数。

在 retriever 检查之前，`train.sh` 会先运行 teacher/student readiness
preflight。它拒绝不存在的 checkpoint、`.merged.in_progress` 或临时分片，
检查 safetensors index 的全部 shard 和 tensor inventory，验证 merged
manifest 中的 100 个来源 checkpoint（`global_step_5..global_step_500`），
并拒绝教师目录与训练输出目录重叠。若 teacher manifest 提供 `sha256`，
该值必须是 `teacher_readiness_opd.directory_sha256` 定义的完整目录摘要。
最终目录名为 `merged`（或带有 `merge_manifest.json`）时，teacher manifest
必须提供已验证的完整目录 `sha256`；普通 temporal checkpoint 可省略它。
`OPD_COMPUTE_TEACHER_DIGEST=1` 可在启动时额外输出摘要。使用 manifest
启动时，脚本还会把 manifest 的文件摘要传入配置阶段，若 preflight 后清单被
替换会直接失败；合并未完成时不会进入 Hydra、Ray 或训练流程。

## 文件职责

- `core_opd.py`：采样 reverse-KL、等权多教师聚合和配置无关的 tensor 逻辑。
- `dp_actor_opd.py`：复用原 PPO forward/optimizer；逐教师计算
  stop-gradient PG-OPD loss，再显式 `sum / teacher_count`。
- `fsdp_workers_opd.py`：复用原 FSDP worker；reference worker 只加载一个
  冻结教师，验证后按原子状态文件切换，并返回
  `[batch, response_length, 1]` 的 log-prob。
- `teacher_readiness_opd.py`：只读检查最终教师、merged 完成清单、权重索引、
  safetensors inventory、student/teacher 接口和输出路径隔离。
- `preflight_opd.py`：`train.sh` 调用的启动前 readiness 命令行入口。
- `generated_token_mask_opd.py`：验证多轮 trajectory 的 generated-only
  mask，确保 observation/padding 不进入 OPD loss。
- `checkpoint_retention_opd.py`：按 Bamboogle/Musique/TriviaQA 三个测试集
  等权平均 EM 计算最佳检查点。
- `checkpoint_trainer_opd.py`：复用原训练器的 rollout/验证/export，仅保留一个
  三测试集均值最高的 HF actor checkpoint，并在每次 validation 后发布教师。
- `main_ppo_opd.py`：独立 Ray/Hydra 入口，复用原 Search-R1 数据和 rollout。
- `config_opd.py`：解析确定性教师 manifest、校验关键约束并写运行清单。
- `train.sh`：唯一启动入口。

## 检查点选择

每个 `SAVE_FREQ` step 都先完成 NQ、HotpotQA validation 和配置的三个
test evaluation。checkpoint 选择分数严格定义为：

```text
(Bamboogle EM + Musique EM + TriviaQA EM) / 3
```

三个数据集等权，避免单个数据集主导选择。分数更高的候选会先
原子发布，再删除旧的 checkpoint；同分时保留较新的 step，因此
`CHECKPOINT_DIR` 下始终最多有一个 `global_step_*` 目录。所有候选的分数和
选择结果写入 `checkpoint_selection.json` 与
`validation_checkpoint_metrics.jsonl`，便于复现和审计。

## 自适应教师调度

启动时先读取每个教师在 NQ/HotpotQA 上的 solved UID 集合。若某个集合是
另一个 checkpoint 集合的子集（包括相等集合），该 checkpoint 会从 active
候选中剔除，并在运行清单和 `adaptive_teacher_selection.json` 中记录原因。

每次学生完成 validation，令学生 solved 集合为 `P`、教师集合为 `Qi`：

- 每道题的初始权重为 1；
- 对所有 active 教师的 `Qi - P` 取并集，每轮权重加 1，最高为 5；
- 教师优先级是 `sum(weight(question) for question in Qi - P)`；
- 选择最高优先级教师，状态文件保存权重、完整 ranking、active/pruned
  清单和历史。

同一题被多个教师覆盖时，一轮只增加一次权重，避免教师清单顺序影响结果。

当前实现是 sampled-token PG-OPD，不是需要完整 vocabulary logits 的
forward-KL/GKD。不同 tokenizer、LoRA ref-in-actor、Megatron worker 和
多教师质量路由不在本目录的实现范围内。
