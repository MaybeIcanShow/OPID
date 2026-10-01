# ToolBench GRPO 四卡训练提速与瓶颈总结

本文记录 2026 年 10 月 1 日对 Qwen3 1.7B ToolBench GRPO 的启动配置调整，解释提速来源、一个训练步的实际工作量，以及 GPU 利用率高时仍然耗时的原因。结论基于当前已完成的 **step 2**、历史运行日志、进程级 GPU 采样和本地实现；尚未做逐项消融实验。

运行状态更新：13:31 已在 GPU `1,2,5` 从原始权重启动多轮协议修复版，默认使用 MirrorAPI-Cache 工具模拟、MirrorAPI 评分。12:43 版本已停止；原因、对照验证和当前配置见文末“多轮生成协议二次修复”。下文历史四卡性能及旧质量指标保留为记录。

**主要收益来自把逐条计算改成批量计算，并减少训练外的验证开销。** 当前一步耗时 **678.6 秒，即 11.3 分钟**。相对下文历史参考的 2029.7 秒，观测耗时比为 **2.99 倍**；扣除历史步中的验证耗时后，比值为 **2.57 倍**。这两个比值都有比较条件限制，不能作为严格的同批次基准结果。

## 比较口径与实测结果

历史文件中拼接了多次运行，必须先区分来源。

- 本文逐阶段比较采用历史 `timing_s/step:2029.664` 的 step 1。该步完成后，这次旧运行在后续更新中发生 OOM；它使用 GPU `1,2,4,5`，optimizer offload 为 `False`。
- 实际产生续训检查点的最后一次旧运行已启用 optimizer offload，日志末尾显示约 `34:01`，但没有完整分阶段计时，不能直接拿它填入下面的阶段表。
- 当前运行使用 GPU `0,1,2,5`，从 `global_step_1` 恢复，完成了 step 2。optimizer offload 为 `True`，这一新步没有执行验证或保存检查点。
- 旧、新两步都是四卡、batch 16、group 8、actor minibatch 16；处理的有效 token 分别为 **1,207,666** 和 **1,181,708**，新步少约 **2.15%**。批次内容、生成长度、物理 GPU、其他用户负载及 offload 配置存在差异。

以下单位均为秒，倍数为旧耗时除以新耗时。

| 阶段 | 历史参考 step 1 | 当前 step 2 | 观测耗时比 | 占当前整步 |
| --- | ---: | ---: | ---: | ---: |
| 多轮采样与环境交互 | 266.8 | 239.8 | 1.11 倍 | 35.3% |
| Actor old log prob | 240.6 | 53.6 | 4.49 倍 | 7.9% |
| Reference log prob | 232.5 | 51.2 | 4.54 倍 | 7.5% |
| Actor 参数更新 | 999.8 | 332.3 | 3.01 倍 | 49.0% |
| 验证 | 288.4 | 本步未执行 | 不适用 | 0% |
| 奖励计算与轨迹写盘等剩余开销 | 约 1.5 | 约 1.6 | 不适用 | 约 0.2% |
| **整步** | **2029.7** | **678.6** | **2.99 倍** | **100%** |

这一步少用的约 **1351 秒**，按阶段差额分解：actor 更新减少约 **668 秒**，两次 log prob 合计减少约 **368 秒**，本步不验证减少约 **288 秒**，采样减少约 **27 秒**。这只是耗时差额分解，不是各个配置项的独立因果贡献。

旧日志中还有一次 **1649.0 秒**的 step 1，但 actor minibatch 为 32，也不应混为相同配置的基线。本次只有一个完整步用于上述比较，长期均值还会受到每 10 步验证和保存的影响。首次模型加载、数据预处理及 vLLM 编译发生在训练步之外，不计入 678.6 秒。

证据：[历史混合日志](../../grpo_qwen3_1.7b_toolbench_2048.log)、[当前训练日志](../../outputs/grpo_toolbench_4gpu_20261001_014954/train.log)、[首步验证记录](../../outputs/grpo_toolbench_4gpu_20261001_014954/verification.json)。

## 实际做了哪些调整

新增了[四卡启动包装脚本](../../examples/grpo_trainer/run_toolbench_qwen3_4gpu.sh)，调用用户指定的[原始启动脚本](../../examples/grpo_trainer/run_toolbench_qwen3.sh)，通过末尾 Hydra 参数覆盖配置。没有改写 GRPO 算法、奖励函数或模型结构。

| 调整 | 历史参考配置 | 当前配置 | 作用与证据边界 |
| --- | --- | --- | --- |
| Actor microbatch | 固定每卡 1 条，动态批次关闭 | `actor.use_dynamic_bsz=True`，token 预算 8192 | 同一次 forward/backward 可合并多条短样本，减少小批次执行和通信次数 |
| Old log prob 和 reference | 固定每卡 1 条 | 两者动态批次均开启，token 预算均为 8192 | 可以对整卡样本按长度打包；对应两个阶段由合计 473.0 秒降至 104.9 秒 |
| 熵计算 | `actor.use_torch_compile=False` | `True` | 本项目只编译 `entropy_from_logits`，有利于融合运算、减少中间结果和调度开销；并非把整个 FSDP 模型编译了 |
| vLLM 执行 | `enforce_eager=True`，chunked prefill 关闭 | `enforce_eager=False`，chunked prefill 开启 | 启用 CUDA Graph 和分块预填充；图捕获批量为 `[1,2,4,8,16,32]` |
| vLLM 显存预算 | `gpu_memory_utilization=0.35` | `0.40` | 给推理引擎更多显存空间；这是显存预算比例，不是 GPU 计算利用率目标 |
| 验证 | 每步验证，启动前验证开启 | 每 10 步验证，启动前验证关闭 | 减少训练外工作；最后一步仍按训练器逻辑验证。跳过启动前验证只影响启动耗时 |
| 检查点保存 | 原脚本默认每步；上述历史参考实际每 10 步 | 每 10 步 | 对原默认值减少写盘频率；历史参考和当前比较步都未保存，不能给本次阶段提速归因 |
| 优化器卸载 | 历史参考为 `False`；检查点来源运行已为 `True` | 保持 `True` | 为共享卡留出显存，是稳定性取舍；CPU/GPU 搬运本身有成本 |
| CPU 与运行时 | Ray CPU 配额 16 | Ray CPU 配额 32，默认 `OMP_NUM_THREADS=4` | 为数据及环境处理提供资源；没有独立测出这项的贡献 |
| 分配器兼容 | 未作为性能对照单独测量 | `expandable_segments=False` | 避免与当前 vLLM sleep mode 内存池冲突；属于兼容配置 |

**四卡和 FlashAttention 在历史运行中已经启用，因此不能把约三倍的观测提速归因于本次才增加 GPU 或才启用 FlashAttention。** 本次确认了 FSDP 使用 `flash_attention_2`、四个 vLLM worker 使用 `FLASH_ATTN`，并在指定四张卡上实际通过 FlashAttention BF16 前向和反向测试。

同样，remove padding、梯度检查点、group 8、max steps 12、最大响应 1024 和 thinking 模式都已存在，本次没有将它们作为新增优化。首轮 vLLM 编译约 54–56 秒，日志中的图捕获额外显存约 0.10 GiB/卡；这些是初始化开销。

源码证据：[动态批次与熵计算](../../verl/workers/actor/dp_actor.py)、[模型与 offload 实现](../../verl/workers/fsdp_workers.py)、[vLLM 参数传递](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py)。多个优化同时启用，若要确认每项的净收益，需要后续逐项消融。

## 为什么一个训练步仍然需要十多分钟

这里进度条上的一个 step 是一整个 GRPO 采样和更新周期，不是一次普通的梯度更新。

```text
16 个任务 × 每个任务 8 条轨迹 = 128 条轨迹
128 条轨迹 × 本步每条 12 轮 = 1536 个回合训练样本
1536 个样本 ÷ 4 张卡 = 每卡 384 个样本
全局 actor minibatch 16 ÷ 4 张卡 = 每卡每次 4 个样本
每卡 384 ÷ 4 = 96 次全局同步 optimizer update
```

本次不仅是按上限估算：[step 2 轨迹文件](../../outputs/grpo_toolbench_4gpu_20261001_014954/rollouts_before_restart/2.jsonl)实际有 **1536 行、128 条轨迹、16 个任务**，回合编号为 0–11，日志的 episode 长度最小值、平均值和最大值均为 12。一般情况下提前结束的轨迹会减少工作量。

每个样本平均有 **167.7 个 prompt token 和 601.6 个 response token**。这一批累计约 **118 万个有效 token**，其中生成响应约 **92 万 token**。这些样本随后还需要 old policy 前向计算、reference 前向计算，以及 actor 的前向、反向、梯度同步与优化器更新。当前未启用 critic 网络训练；日志中的 `critic/score` 等指标名称不代表另有 critic 更新耗时。

还有一个容易误解的配置细节：**8192 token 预算没有把 actor minibatch 扩大到 8192 token。** actor 先切成每卡 4 条的 minibatch，再在其中划分 microbatch。因此单次 actor forward 最多合并 4 条，按本步均值大约只有 **3077 token**。长样本仍可能需要拆分，跨卡还要对齐 microbatch 数量。

相比之下，old/reference log prob 可以在每卡 384 条样本范围内进行动态打包，不受上述 4 条上限限制。这解释了为什么这两个阶段的改善特别明显。

实现依据：[轨迹展开和有效回合收集](../../agent_system/multi_turn_rollout/rollout_loop.py)、[minibatch 按卡数归一化](../../verl/workers/fsdp_workers.py)、[每个 minibatch 执行一次 optimizer step](../../verl/workers/actor/dp_actor.py)。

## 为什么 GPU 很满仍然会慢

需要区分三种不同的指标。

| 指标 | 表示什么 | 本次如何解读 |
| --- | --- | --- |
| `nvidia-smi` 的 GPU utilization | 采样期间至少有一个 GPU kernel 在执行的时间比例 | 100% 不等于模型达到了峰值 FLOPs；通信、访存及其他进程也可能贡献高读数 |
| 进程级 `pmon` 的 SM 指标 | 按进程归属的采样利用率 | 比整卡读数更能说明 GRPO 自身占用，但仍不能分解计算与通信 |
| 日志 `perf/mfu/actor` | 按模型计算量、耗时和设备峰值估算的模型算力利用率 | 本步为 **0.097，即约 9.7%**；它是公式估算，未单独计入梯度检查点增加的重计算，不是硬件 profiler 的测量 |

在 old log prob 阶段连续约 30 秒的 7 次进程级采样中，本次 GRPO 的结果为：

| 物理 GPU | GRPO 进程 SM 平均读数 | 当时的资源情况 |
| --- | ---: | --- |
| 0 | 92.9% | 其他进程主要占少量显存 |
| 1 | 41.7% | 已有其他用户计算任务，约占 17 GB 显存 |
| 2 | 52.3% | 已有其他用户计算任务，约占 18 GB 显存 |
| 5 | 92.4% | 其他进程主要占少量显存 |

这是一个阶段的短窗口，不是整步或整场训练的平均利用率。共享卡的总利用率接近 100%，也不能全部计入本任务。

训练采用四卡同步 FSDP，每个更新需要跨卡通信；多轮采样也要等待这一轮各 worker 返回，才能推进环境。1、2 号卡与其他作业竞争算力，能够解释为什么 0、5 号卡在生成活跃时利用率很高，却会在部分时段空闲。当前证据支持资源竞争和同步等待确实存在，但没有 profiler 数据可以量化它们各占 332 秒更新耗时的多少。

此外，显存占用也不是计算效率。vLLM KV cache、模型与优化器状态、激活、CUDA Graph 和 PyTorch 缓存都占显存。日志中的 `max_memory_reserved_gb=59.713` 超过本机单卡容量，不能直接解释为某一张卡的物理显存峰值；应结合其统计与内存池口径以及 NVML 读数判断。此前观察到的是进程显存随阶段上升、rollout sleep 后释放，首个新步完成时没有 OOM。

证据：[进程级原始采样](../../outputs/grpo_toolbench_4gpu_20261001_014954/gpu_sample_training.txt)、[采样汇总](../../outputs/grpo_toolbench_4gpu_20261001_014954/gpu_training_summary.json)、[MFU 计算](../../verl/utils/flops_counter.py)。

## 当前瓶颈及后续可验证的方向

从已测出的阶段时间看，优先级是 **actor 更新 49.0% → 多轮采样 35.3% → 两次 log prob 合计 15.5%**。这三部分合计约占整步的 99.8%。

**Actor 更新是最大的已观测耗时项。** 当前每卡仅 4 条的 minibatch 带来 96 次同步更新，仍有较多小批次和通信开销。梯度检查点带来重计算，大词表 logits 与熵计算也有显存和访存成本。优化器在整个 `update_actor` 前后各搬运一次，并不是在 96 个 minibatch 中每次都搬；reference 模型在当前 FSDP1 实现中还被代码设为 CPU offload。这些成本的独立占比尚未测量。

**采样是第二大耗时项。** thinking 模式、平均约 602 token 的响应、128 条轨迹和全部跑满 12 轮，使解码工作量很大，且同一条轨迹的下一轮依赖上一轮环境反馈。CUDA Graph 可以减少调度开销，但不能消除自回归生成和跨轮依赖。本步成功率为 0、有效 action 比例约为 51.6%，因此还值得检查输出协议与任务完成行为；提速并不代表任务效果已改善。

`timing_s/gen` 包含整个轨迹收集函数，也包含环境交互与等待。单独的 `timing_s/reward=0.441` 只是后续奖励处理时间，不能据此断言工具服务调用只用了 0.4 秒，或已经证明网络不是瓶颈。当前 ToolBench 优先使用轨迹内响应和磁盘缓存，HTTP 是后备；本次没有新增外部工具服务。

下列是后续实验方向，本次没有继续修改正在运行的配置：

| 优先级 | 要验证的问题 | 可执行的后续实验 | 需要保持的比较条件 |
| --- | --- | --- | --- |
| 高 | 共享卡竞争和同步等待占多少 | 在四张独占 GPU 上复测，或对当前运行做阶段 profiler | 保持同样模型、任务批次、生成设置及 batch |
| 高 | 每卡 actor minibatch 4 是否限制吞吐 | 试验全局 minibatch 32、64，并分别监控显存和质量 | 会改变每个 outer step 的 optimizer 更新次数与优化行为，不能仅按速度选值 |
| 中 | rollout 时间花在解码还是环境等待 | 增加每轮生成、环境 step、权重切换的独立计时 | 环境等待、tokenizer 和解码分别统计 |
| 中 | 长响应和跑满 12 轮能否改善 | 先检查 action 格式、Finish 行为及任务成功率 | 缩短响应、减少轮数或关闭 thinking 会改变任务行为，需单独评估 |
| 中 | offload 与重计算是否值得保留 | 有充足显存后分别对 offload、gradient checkpointing 做消融 | 当前共享卡空间有限，历史已有 OOM，不能同时盲目关闭 |

单纯继续提高 vLLM 显存比例，主要影响推理阶段缓存空间，不能直接解决占约一半时间的 actor 更新。也不能只看 GPU utilization 是否达到 100% 来选择配置。

## 空响应异常原因与恢复记录

2026 年 10 月 1 日 02:16，原运行已完成 step 3，在 step 4 的多轮采样中退出。直接异常是 `TypeError: object of type 'NoneType' has no len()`，发生于 `ToolBenchMultiProcessEnv._step_one` 对离线工具响应列表调用 `len(responses)`。本次退出没有出现 CUDA OOM 报错。

空值来自 **Arrow 和 Parquet 对动态字典的表示方式**。预处理使用 `Dataset.from_list` 将每条记录的 `offline_tool_responses` 字典转换为 Arrow 数据，再写入 Parquet。不同样本包含的工具键不相同，Arrow 会把这些键合并成一个 struct 的字段集合；回读时，当前样本原本不存在的字段表现为 `None`。

例如，两条原始记录分别为：

```python
# 写入前，每个样本只包含自己的工具。
row_a = {"offline_tool_responses": {"weather": ["weather result"]}}
row_b = {"offline_tool_responses": {"search": ["search result"]}}

# Arrow 和 Parquet 回读后的表示。
row_a = {"offline_tool_responses": {"weather": ["weather result"], "search": None}}
row_b = {"offline_tool_responses": {"weather": None, "search": ["search result"]}}
```

如果第一条记录生成了 `Action: search`，原来的代码会取到 `None`：

```python
responses = record.get("offline_tool_responses", {}).get(name, [])
response = responses[used] if used < len(responses) else None
```

`.get(name, [])` 的默认值只在键不存在时生效；`"search": None` 的键存在，所以不会返回空列表。模型生成不同工具名时，这个已有数据表示问题才会触发，解释了为什么前几步正常而第 4 步报错。

实际训练 Parquet 中，每行的响应 struct 有 2218 个联合工具字段；检查的第 0 行有 2216 个空字段。完整 2048 行中，共有 **4,539,127 个此类空字段**。它们不代表有这么多独立样本或真实工具服务失败，而是稀疏字典展开后的缺失值。使用真实 Parquet 的空字段能够复现同一异常。

修复是在原有响应读取后明确处理 `None`：

```python
responses = record.get("offline_tool_responses", {}).get(name, [])
if responses is None:
    responses = []
```

这样，无可用离线响应时会继续走原有的磁盘缓存、HTTP 后备及 missing response 处理流程；全部不可用时仍返回原来的 `-0.05` 奖励。没有伪造工具响应或改变奖励定义。采用显式 `is None` 判断，也兼容回读后可能出现的 NumPy 响应数组，避免 `responses or []` 对多元素数组触发真假值歧义。

两项针对性回归检查已通过：一项实际写入并回读稀疏字典 Parquet，验证空响应不会中断同批其他轨迹；另一项验证 NumPy 响应数组仍按顺序消费，耗尽后走原有后备流程。当前环境未安装 pytest，检查通过直接调用两项测试函数及其断言执行，没有为此安装依赖。

恢复时只清理了该已退出任务遗留的四个 worker，保留其他用户进程；02:21 按原四卡优化配置重新启动。最新持久化检查点仍是 step 1，因此 step 2、3 需要重跑。每 10 步保存和验证的设置保持不变，原 step 2、3 的轨迹已归档。

- [空响应处理修复](../../agent_system/environments/env_package/toolbench/envs.py)
- [回归检查代码](../../tests/test_toolbench_env.py)
- [预处理代码](../../examples/data_preprocess/preprocess_toolbench.py)
- [故障日志](../../outputs/grpo_toolbench_4gpu_20261001_014954/train.log)，可检索 `TypeError: object of type 'NoneType' has no len()`
- [恢复运行日志](../../outputs/grpo_toolbench_4gpu_20261001_022105/train.log)
- [恢复运行记录](../../outputs/grpo_toolbench_4gpu_20261001_022105/launch.json)

## 从原始权重重新开始

按用户要求，2026 年 10 月 1 日 02:27 已停止上一份续训任务，并用修复后的代码从原始模型重新启动。模型路径为 `/home/zhengbaowei/model/Qwen3-1.7B`；配置强制使用 `trainer.resume_mode=disable` 和 `trainer.resume_from_path=null`，跳过所有旧检查点加载。优化器、调度器、数据进度和 global step 全部重新初始化，进度从 0/128 开始，第一个训练步记作 step 1。

GPU 仍是 0、1、2、5，保留 FlashAttention、动态批次与 CUDA Graph；仍每 10 步保存和验证。旧检查点及日志保留，新权重保存到独立目录：

```text
/home/zhengbaowei/model/ckpt/grpo_qwen3_1.7b_toolbench_2048_fresh_20261001_022739
```

- [从原始权重启动的脚本](../../examples/grpo_trainer/run_toolbench_qwen3_4gpu_fresh.sh)
- [新训练日志](../../outputs/grpo_toolbench_4gpu_fresh_20261001_022739/train.log)
- [新训练启动记录](../../outputs/grpo_toolbench_4gpu_fresh_20261001_022739/launch.json)

该启动脚本为每次运行创建独立名称，若指定输出目录已经存在则拒绝启动，避免覆盖旧实验。

## TensorBoard 训练面板

2026 年 10 月 1 日 02:34 已启动 TensorBoard，地址为 <http://127.0.0.1:6006/>。远程使用 VS Code 时，在“端口 / Ports”中转发服务器端口 `6006`，然后打开转发后的本地地址。服务只绑定服务器回环地址，不占用训练 GPU。

当前展示的实验为 `qwen3_1.7b_toolbench_fresh_20261001_022739`，对应本次从原始权重启动的训练。在 **Custom Scalars** 页配置了训练损失、奖励与任务效果、优化稳定性、轨迹长度、训练性能、验证六组面板；**Text** 页记录本次运行配置，**Scalars** 页可检索全部原始指标。

训练在每个完整 outer step 结束时集中写入指标，因此 step 1 完成前曲线为空属于正常情况；现有速度约每 11 分钟产生一个点，事件刷盘最多再延迟约 2 分钟。验证图从 step 10 开始更新，之后每 10 步及最后一步更新。可以按 Step 或 Wall time 查看，分析波动时建议先将 smoothing 设为 0。

面板沿用训练程序已有的真实指标。`actor/entropy_loss` 图表示更新前策略的平均熵，MFU 是 0–1 比例的估计值，吞吐单位为有效 token/s/GPU。训练当前没有独立记录 total loss。

- [TensorBoard 启动脚本](../../scripts/run_toolbench_tensorboard.sh)
- [服务记录与 PID](../../outputs/tensorboard/service.json)
- [服务日志](../../outputs/tensorboard/server.log)
- [定制面板配置](../../outputs/grpo_toolbench_4gpu_fresh_20261001_022739/tensorboard_dashboard.json)

实际事件目录为本次权重目录下的 `tensorboard/`，通过 `outputs/tensorboard/runs/` 中的实验名称软链接注册。服务启动时启用了多事件文件持续读取，可同时读取训练写入和面板配置。现有训练无需重启。

若以后需要重新启动已停止的 TensorBoard 服务，可在项目根目录执行以下命令；当前服务正在运行，无需重复启动。

```bash
bash scripts/run_toolbench_tensorboard.sh
```

## 迁移到 GPU 1、2、5、7

按用户要求，已停止原 `0,1,2,5` 实验的专属进程会话，保留其他进程。从原始模型开始训练的实验已完成 step 52，最近完整持久化检查点为 step 50；迁移后从此检查点恢复模型、优化器、调度器与数据进度，重跑未持久化的步骤。旧权重、rollouts 和 TensorBoard 事件保留，新运行另建目录。

用户确认的显存口径为：**GPU 7 给其他任务合计预留约 24 GiB 容量，本实验最多约 20 GiB**。这不是要求所有任务运行后仍有 24 GiB 空闲，也不是硬件隔离或分配器硬上限，需要用本实验进程的实际占用验证。

| 参数 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 物理 GPU | 0,1,2,5 | 1,2,5,7 |
| vLLM 显存比例 | 0.40 | 0.25 |
| Actor 更新微批次 | 动态 token 预算 8192 | 固定每卡 1 条 |
| Old/ref log prob | 动态 token 预算 8192 | 动态 token 预算 5120 |
| Actor 参数 offload | False | True |
| 优化器 offload | True | True |
| vLLM max model len / batched tokens | 8192 / 8192 | 5120 / 5120 |
| vLLM max num seqs | 128 | 64 |

保留 FlashAttention、CUDA Graph、梯度检查点、train batch 16、group 8、全局 actor minibatch 16、prompt 4096、response 1024、每 10 步保存和验证。固定单样本微批次会降低参数更新吞吐，优先满足共享 GPU 的显存预算；迁移前约 9–11 分钟的单步速度不能直接沿用。

动态 token 预算不能直接降为 4096：本地实现要求预算不小于填充后的序列宽度 `4096 + 1024 = 5120`。vLLM 的本地封装也要求启用 chunked prefill 时 batched tokens 不小于 model len，所以两者设为 5120。

- [迁移续训启动脚本](../../examples/grpo_trainer/run_toolbench_qwen3_1257_resume.sh)
- [当前四卡参数](../../examples/grpo_trainer/run_toolbench_qwen3_4gpu.sh)
- [迁移后的日志](../../outputs/grpo_toolbench_gpu1257_20261001_112912/train.log)
- [迁移后的启动记录](../../outputs/grpo_toolbench_gpu1257_20261001_112912/launch.json)
- [各卡进程显存采样](../../outputs/grpo_toolbench_gpu1257_20261001_112912/gpu_memory_samples.jsonl)

TensorBoard 地址仍为 <http://127.0.0.1:6006/>，选择实验 `qwen3_1.7b_toolbench_gpu1257_20261001_112912`。新曲线从恢复后的 step 51 开始，旧实验曲线可同时选择对比。

11:39 已验证从 step 50 恢复并进入 step 51 的反向更新。覆盖初始化、检查点加载、生成、old/ref log prob 和已执行的反向更新部分，GPU 7 的本实验进程采样峰值为 **17.8 GiB**，GPU 0 的本实验占用始终为 0。该值来自每 5 秒采样，首个恢复步尚未完整结束，不代表后续所有步骤的硬上限。详见[迁移验证记录](../../outputs/grpo_toolbench_gpu1257_20261001_112912/verification.json)。

## 复现与记录位置

从原始权重重新建实验的入口如下。本节命令对应历史四卡实验；当前三卡 StableToolBench 实验入口见文末，正在运行时无需重复启动。

```bash
cd /home/zhengbaowei/paper_project/OPID
bash examples/grpo_trainer/run_toolbench_qwen3_4gpu_fresh.sh
```

配置保留了 batch 16、group 8、actor minibatch 16、每条最多 12 轮、最大 prompt 4096、最大 response 1024、梯度检查点和原 GRPO 损失。每 10 步验证和保存，最后一步也会执行相应操作。

- [启动配置与进程记录](../../outputs/grpo_toolbench_4gpu_20261001_014954/launch.json)
- [训练日志](../../outputs/grpo_toolbench_4gpu_20261001_014954/train.log)
- [本报告对比数据](toolbench_grpo_4gpu_performance_evidence.json)
- [训练 step 的计时边界](../../verl/trainer/ppo/ray_trainer.py)
- [吞吐指标定义](../../verl/trainer/ppo/metric_utils.py)

日志中的 `perf/throughput=435.337` 按 `有效 token 数 ÷ 整步秒数 ÷ GPU 数` 计算，单位是 **有效 token/s/GPU**，不是纯生成速度，也没有把重复前向和反向计算逐次累加成 token 数。

## 训练效果与验证问题记录

记录日期：2026 年 10 月 1 日。以下质量分析基于从原始权重启动的实验 `grpo_toolbench_4gpu_fresh_20261001_022739` 的 step 1–52、step 10–50 的五次验证，以及当日检查的代码和数据。这是该实验的检查快照，不代表迁移续训后的最新结果。以下是修复前已确认的问题快照，与前文的 Parquet 空值异常不同。后续已按用户要求实施修复，当前配置和验收记录见文末“StableToolBench 修复与 GPU 1、2、5 全新运行”。

**输出格式明显改善，但当前指标不足以证明真实任务完成能力提升。** Action 可解析率由 step 1 的 33.98% 升至 step 52 的 99.83%，平均轨迹奖励由 −0.5996 升至 −0.4008；step 52 的规则成功率为 3/128 条训练轨迹，并不代表 3 个唯一任务。已记录指标没有 NaN/Inf，但环境上下文、工具反馈及评分存在下列问题。

### 多轮交互丢失原始任务和历史

**证据：** `ToolBenchEnvironmentManager.step` 只返回当前工具响应，未把初始问题、工具说明和历史动作带入下一轮。step 50、52 保存的所有后续轮次输入都不再含原始 `User query` 和工具列表。例如 `52.jsonl` 第 3 行的完整输入只有：

```text
user
Tool response (missing):
{"error": "tool response unavailable", "response": ""}

Continue with one Thought/Action/Action Input step.
assistant
```

**影响：** 模型无法根据完整任务和已执行步骤继续规划；实际输出出现解释工具报错、请求更多上下文及无效重试。当前 ToolBench manager 没有实现历史拼接，不能仅通过增大 `env.history_length` 认定问题已解决。

**后续验收：** 检查第二轮及后续实际模型输入，确认保留原始任务、工具定义、历史动作及对应响应，并明确长上下文截断策略。

来源：[环境管理器](../../agent_system/environments/env_manager.py) 的 `ToolBenchEnvironmentManager.step`（检查时第 821 行）；[step 52 训练轨迹](/home/zhengbaowei/model/ckpt/grpo_qwen3_1.7b_toolbench_2048_fresh_20261001_022739/rollouts/52.jsonl)。

### 工具响应大量不可用

**证据：** step 52 共 128 条训练轨迹、1184 条逐轮记录。其中 96 条轨迹跑满 12 轮，其余 32 条首轮终止。1056 条后续输入中，988 条为 `Tool response (missing)`，占 **93.56%**；66 条为离线响应，2 条为动作格式错误。step 50 后续输入中的 missing 比例为 1137/1224，即 92.89%。这些比例统计的是**训练轨迹后续输入**，不是验证集失败率，也不是所有工具请求的精确失败率。

响应查找顺序为“示范轨迹离线响应 → 磁盘缓存 → HTTP 服务 → missing”。当日约 11:35 检查时，配置的 `http://127.0.0.1:12001/virtual` 没有监听服务；这只是当时的服务状态，不能推断整个训练期间都未启动。磁盘缓存另有下一节所述路径问题。缺少工具映射、参数未命中等也可能造成 missing，尚未逐项量化其贡献。

**影响：** 大量交互得到相同错误反馈和每次 −0.05 的环境奖励。上下文丢失和工具不可用共同限制任务完成，但现有检查没有测出各自的独立影响。

**后续验收：** 分别记录缓存命中、参数未命中、未知工具、服务错误和响应来源；在固定样本上验证工具响应与请求参数对应，并明确缓存未命中时的处理方式。

来源：[响应回退逻辑](../../agent_system/environments/env_package/toolbench/envs.py) 的 `_step_one`（检查时第 87 行起）；[step 50 训练轨迹](/home/zhengbaowei/model/ckpt/grpo_qwen3_1.7b_toolbench_2048_fresh_20261001_022739/rollouts/50.jsonl)；上述 step 52 轨迹。

### StableToolBench 缓存路径不兼容

**证据：** 当前 `_cache_response` 将 category 全部转为小写，tool 目录也未补官方缓存使用的 `_for_Category` 后缀。例如代码查找 `medical/getguidelines/vac.json`，实际文件为 `Medical/getguidelines_for_Medical/vac.json`。

对 128 条验证样本中的 756 个非空工具映射检查：按现有代码拼接路径，文件存在数为 **0/756**；按官方目录规则处理 category 大小写、tool 后缀和 API 名后，可找到 **620/756（82.0%）**。后者只是按映射计数的 **API 文件覆盖率**，不是实际缓存命中率；输入参数还必须匹配缓存键。

此外，现有离线回放只按工具名和调用次数取示范响应，忽略实际 `action_input`，不能代替按请求参数匹配的 StableToolBench 缓存。

**影响：** 已配置缓存目录不等于已有效使用缓存；当前查找规则使这批验证工具映射无法找到已有缓存文件。

**后续验收：** 与官方路径和请求键规则对齐，用真实缓存请求验证命中及参数不匹配时的行为；验证流程应避免以忽略参数的示范回放替代真实工具执行。

来源：[当前缓存读取](../../agent_system/environments/env_package/toolbench/envs.py) 的 `_cache_response`（检查时第 40 行起）；[官方缓存服务实现](https://github.com/THUNLP-MT/StableToolBench/blob/master/server/main.py)。

**资源结论：** 本地 `data/StableToolBench/tool_response_cache` 有 12255 个 JSON 文件，合计约 466.8 MiB。文件读取、JSON 解析及键查询使用 CPU、系统内存和磁盘，缓存本身不需要显存。这不包含待评测策略模型的推理开销，也不包含另行部署的工具模拟模型或答案评测模型；官方 `MirrorAPI-Cache` 是模型名称，不能与这些响应缓存文件混为一谈。

### 成功判断未验证任务完成

**证据：** 环境把 `Finish` 且 `return_type == "give_answer"` 直接记为成功并奖励 +1；JSON 解析失败时仅检查字符串是否包含 `give_answer`。代码未判断最终答案是否正确、是否覆盖任务要求。

step 52 的三条计为成功的**训练样本**均在首轮直接 Finish，未调用任务工具：

| 轨迹文件位置 | 最终答案内容 |
| --- | --- |
| `52.jsonl` 第 98 行 | 请求提供园艺论坛 URL，未返回所需植物列表 |
| `52.jsonl` 第 343 行 | 声称无法取得泰国行政区数据，建议用户查询官方来源 |
| `52.jsonl` 第 1136 行 | 请求用户提供星座 |

这些训练样本未完成原始任务。验证使用相同的成功规则，但**没有逐条核验验证集的三条成功输出，不能将上述训练案例当作验证案例**。Action validity 也仅表示能解析出非空工具名，不检查工具存在性、参数合法性或执行成功。

**影响：** 当前 `success_rate` / `toolbench_success_rate` 是结束动作的规则指标，不能作为真实答案正确率或 StableToolBench 正式评测成绩；奖励存在鼓励请求澄清或无法完成时直接结束的空间。

**后续验收：** 接入选定版本的 StableToolBench 评测流程，记录评测集、评测器和配置；把动作格式、工具调用结果、结束状态及任务完成判定分开记录，保留最终答案和评测结果。上述未完成案例不应仅凭 Finish 获得任务成功分。

来源：[成功判定](../../agent_system/environments/env_package/toolbench/envs.py) 的 `_step_one`（检查时第 77 行起）；[动作解析指标](../../agent_system/environments/env_package/toolbench/projection.py) 的 `toolbench_projection`；上述 step 52 轨迹。

### 训练集与验证集存在重复任务

**证据：** 当前预处理对原始 `toolllama_G123_dfs_train.json` 按顺序取前 2048 条训练、随后 128 条验证，没有按任务或 query 去重。

| 项目 | 数量 |
| --- | ---: |
| 训练行数 | 2048 |
| 训练唯一 query 数 | 2016 |
| 验证行数 | 128 |
| 验证唯一 query 数 | 127 |
| 验证中与训练 query 完全相同的行数 | 4（3.125%） |

两边 record ID 交集为 0，但不同示范步骤标识仍可能对应同一 query，因此仅检查 record ID 不足以排除任务重复。验证集自身另有一对重复 query。以上是原始 query 的完全匹配审计，尚未覆盖改写或语义近似的任务。

**影响：** 验证集并非严格独立，不能据此可靠判断未见任务上的泛化；尚未量化重复对分数的影响。当前 128 条数据也是原始训练文件的切片，不能当作 StableToolBench 官方测试划分。

**后续验收：** 使用明确版本的 StableToolBench 验证任务并检查与训练任务的重叠；保留划分清单，确保标准化 query 或任务标识不跨训练和验证集合重复，并对验证集自身去重。

来源：[预处理划分代码](../../examples/data_preprocess/preprocess_toolbench.py)（检查时第 114 行起）；[数据元信息](../../data/toolbench_processed/metadata.json)；`data/toolbench_processed/train.parquet` 与 `test.parquet`。

### 当前验证结果及使用限制

以下数值读取自该 fresh 实验的 TensorBoard 原始指标；每次验证 128 条任务、每题采样一次。

| Step | 本地验证分数 test_score | 本地规则成功率 |
| --- | ---: | ---: |
| 10 | −0.577882 | 0/128 |
| 20 | −0.573481 | 0/128 |
| 30 | −0.570035 | 1/128（0.78125%） |
| 40 | −0.566240 | 1/128（0.78125%） |
| 50 | −0.548603 | 3/128（2.34375%） |

`test_score` 对按步骤展开的记录直接平均，因此等于 `Σ(轨迹长度 × 轨迹累计回报) / Σ轨迹长度`。它与每条轨迹等权的 `episode/reward/mean` 不是同一口径，且训练额外使用无效动作惩罚，不能直接以两者差值判断泛化差距。

验证启用随机采样（temperature 0.6、top_p 0.95），没有重复评测，也没有本次 fresh 实验同配置的 step 0 验证。成功任务从 1 条到 3 条仅增加两条，结合以上环境和评分问题，不足以证明稳定的任务能力提升。

来源：[训练及验证日志](../../outputs/grpo_toolbench_4gpu_fresh_20261001_022739/train.log)；[验证聚合实现](../../verl/trainer/ppo/ray_trainer.py) 的 `_validate`；[轨迹奖励写入](../../agent_system/reward_manager/episode.py)；[验证采样配置](../../examples/grpo_trainer/run_toolbench_qwen3.sh)。

### 检查时的待处理事项（历史快照）

以下保留最初的验收清单；实施状态见下一节。

- [ ] 保留并验证每轮输入中的原始任务、工具说明和交互历史。
- [ ] 对齐 StableToolBench 缓存路径与请求键，记录实际命中率，并确定未命中处理方式。
- [ ] 恢复可复核的工具执行流程，区分工具错误、服务不可用和任务失败。
- [ ] 使用 StableToolBench 任务及评测器判断答案完成情况，保留评测明细。
- [ ] 对训练和验证任务做重叠检查与去重，保存固定划分。
- [ ] 修复后建立同配置的原始模型基线，再复评检查点；原有结果保留为修复前记录。

## StableToolBench 修复与 GPU 1、2、5 全新运行（12:43）

此前记录的上下文、工具反馈、缓存路径、成功判定和数据重复问题已完成代码修复。历史成绩继续保留，不能与修复后的任务集及评分口径直接比较。

| 问题 | 本次修改与验收 |
| --- | --- |
| 多轮上下文丢失 | 每轮保留原始任务和工具定义，拼接动作与响应。超出 4096 token 时优先移除最早的完整交互；必要时缩短最新交互并添加标记，不截断原始任务。上下文单测、三卡补齐的 CPU 采集端到端测试及真实工具调用检查通过。 |
| 工具响应不可用 | 默认按完整请求参数读取磁盘缓存；未命中调用 MirrorAPI-Cache。取消忽略参数的示范响应回放。逐轮记录响应来源与错误码，区分工具错误、未知工具和服务异常。实际远程调用通过。 |
| 缓存路径不兼容 | 对齐 category 大小写、tool 的 `_for_Category` 后缀及 API 名；JSON 参数按规范化对象精确匹配，避免字符串、数字、布尔值混淆。缓存命中与未命中测试通过。 |
| Finish 被直接判成功 | `Finish(give_answer)` 必须提交非空答案，由 MirrorAPI 判断完成情况。请求补充信息的真实样例得到失败分。评测异常单独记为 error，不把未评分任务当成已评分失败发布完整成功率。 |
| 训练验证重复 | 验证改为固定版本官方 solvable queries 的 128 条子集；训练 2048 条唯一 query，排除全部 764 条唯一官方验证 query。标准化 query 的训练／验证交集为 0。 |

两个服务的职责如下：

| 用途 | 模型 | 服务 |
| --- | --- | --- |
| 工具模拟，磁盘缓存未命中时使用 | MirrorAPI-Cache | `http://10.8.176.56:8001/v1` |
| 最终答案评分 | MirrorAPI | `http://10.8.176.56:8000/v1` |

本地 JSON 响应缓存仅使用 CPU、内存和磁盘，**不额外占用训练 GPU 显存**。远端 MirrorAPI-Cache 和 MirrorAPI 模型各自占用其部署机器的显存。这里按用户指定，使用 MirrorAPI 配合 StableToolBench 的 FAC 提示词进行评分；结果属于该自定义 judge 配置，不能称为官方专用 Evaluator 模型的标准 FAC 成绩。

官方数据与提示词版本固定为 `aa4ed9f4737ad98bd706663f01d63623c3427812`，验证文件读取前检查 SHA-256。六组验证任务分别为 G1_instruction 22、G1_category 22，其余四组各 21。预处理按实际 Qwen3 聊天模板限制初始输入为 3584 token，为后续交互保留空间；训练候选中排除 18 条超长输入，全部官方候选中排除 23 条超长输入后再选取固定子集。最终训练／验证初始输入最大长度为 3576／3507 token。划分与完整任务清单见[数据元信息](../../data/toolbench_stable_processed/metadata.json)。

新的验证明细保存每题的原始问题、最终答案、终止原因、工具调用次数、响应来源和 judge 结果。指标增加按任务等权的 `episode_reward_mean`、judge 评分覆盖率／错误率／成功率及分组结果；只有覆盖率为 100% 时发布整个集合的 judge 成功率，部分评分另报已评分子集结果。原有按步骤加权的 `test_score` 仅保留兼容。

启动前完成 **73 项 unittest**、Python 编译、shell 语法和 diff 空白检查。针对真实部署服务验证了正确答案、拒绝回答的评分和官方验证任务的工具调用；后一项也确认了第二轮仍保留初始上下文。见[真实服务集成检查](../../outputs/stabletoolbench_integration_smoke.json)。测试结束时已有的 multiprocess 清理器发出 Python 3.12 析构警告，测试结果为 OK，进程退出码为 0。

| 本次启动项 | 配置 |
| --- | --- |
| GPU | 1、2、5，三卡 |
| 初始权重 | `/home/zhengbaowei/model/Qwen3-1.7B` |
| 恢复方式 | `resume_mode=disable`，不读取旧检查点 |
| 训练／验证任务 | 2048／128 |
| Train batch／actor minibatch／group | 12／12／8，每步 96 条训练轨迹 |
| Actor microbatch | 每卡 1 条；参数和优化器 offload |
| Prompt／response／model len | 4096／1024／5120 |
| vLLM 显存比例／max seqs | 0.25／64 |
| 验证和保存 | 启动时验证；之后每 10 步及末步验证、保存 |
| 训练轮次 | 1 epoch，运行日志已确认 170 步 |

12:43 新进程已启动；12:46 完成三卡模型初始化并进入 step 0 初始验证，12:56 初始验证结束后进入第 1 步训练采样。已核实本实验 worker 只位于 GPU 1、2、5。初始化与验证阶段采样的单卡本实验进程显存峰值约 11.7 GiB，不代表反向更新峰值；尚无完整新训练步的吞吐数据。详见[启动验证记录](../../outputs/grpo_stabletoolbench_gpu125_fresh_20261001_124313/verification.json)及[显存采样](../../outputs/grpo_stabletoolbench_gpu125_fresh_20261001_124313/gpu_memory_samples.jsonl)。旧 GPU 1、2、5、7 续训实验在完成 step 54 后停止，仅终止了该实验所属进程，旧日志和检查点保留。

- [三卡全新运行入口](../../examples/grpo_trainer/run_toolbench_qwen3_3gpu_fresh.sh)
- [本次启动记录](../../outputs/grpo_stabletoolbench_gpu125_fresh_20261001_124313/launch.json)
- [本次训练日志](../../outputs/grpo_stabletoolbench_gpu125_fresh_20261001_124313/train.log)
- 检查点目录：`/home/zhengbaowei/model/ckpt/grpo_qwen3_1.7b_stabletoolbench_gpu125_fresh_20261001_124313`
- TensorBoard：<http://127.0.0.1:6006/>，实验 `qwen3_1.7b_stabletoolbench_gpu125_fresh_20261001_124313`

复现命令：

```bash
cd /home/zhengbaowei/paper_project/OPID
bash examples/grpo_trainer/run_toolbench_qwen3_3gpu_fresh.sh
```

### 修复后初始验证（step 0）

2026-10-01 12:56 完成原始 Qwen3-1.7B 的 128 条验证任务：

| 项目 | 结果 |
| --- | ---: |
| 提交最终答案 | 56/128（43.75%） |
| 提交答案后成功／失败 | 0／54 |
| 评分输出无法解析 | 2 |
| 未提交答案的协议失败 | 72（64 条达到轮数上限，8 条主动放弃） |
| 有明确任务结论的覆盖率 | 126/128（98.4375%），包含协议失败 |
| 已有明确结论子集的成功率 | 0/126 |
| 工具调用 | 18 |
| 磁盘缓存响应 | 4/18（22.22%） |
| MirrorAPI-Cache 响应 | 14/18（77.78%） |

本次有 2 条评分异常（query ID 394、3723，分别为状态缺失／歧义和格式不符合要求），因此**不发布整个 128 条集合的成功率**；保留 error 和空分数，没有当作已判失败。上表 126 条明确结论不等于 126 次远端评分：其中 72 条因未提交答案直接按协议判失败，54 条由 MirrorAPI 判失败。缓存与远端工具模拟的响应来源已经在实际多轮验证中生效。

[逐题验证结果](/home/zhengbaowei/model/ckpt/grpo_qwen3_1.7b_stabletoolbench_gpu125_fresh_20261001_124313/validation/0.jsonl)保存在实验目录，未纳入 Git。该结果是单次随机采样的原始模型基线；后续检查点仍需用同配置比较，不能直接对比修复前的规则成功率。

## 多轮生成协议二次修复（13:31）

12:43 版本恢复了历史文本，但把原始 assistant 输出整体拼进单条 user 消息。该实现没有正确处理 Qwen3 的思考块和对话角色，因此上一轮的未闭合思考、模型自行编造的 Observation、甚至错误的多步输出会再次进入模型输入。这是运行协议问题，不能把初始低分全部解释为模型能力差或上下文长度不足。

### 诊断证据

对该版本初始验证的 1100 次无效动作逐条分类：

| 原因 | 次数 |
| --- | ---: |
| 含未闭合的 `<think>` | 728 |
| 无法识别单个规范动作 | 223 |
| 参数 JSON 后夹带其他动作、反馈或文字等 | 147 |
| 工具名无效 | 2 |

对解码文本重新编码，728 条未闭合思考中仅 71 条恰为 1024 token，657 条不等于该长度；有 200–400 token 的短输出也带未闭合思考。旧记录没有生成器原始 `finish_reason`，重新编码也不能完全代替原始 token 计数，因此不能将 728 次都算成输出截断。

例如 query 588 的无效输出自行写出了工具反馈和后续动作，还把这些内容当作已发生的历史继续回答。旧解析器拒绝执行这类输出是正确的，但原样回灌又持续污染下一轮输入。另有输出在一次生成中写出多个 Action，或在 JSON 后接 `Finish`，因此不符合单步执行协议。

两条评分异常也已重放：query 394 的 MirrorAPI 回复先给 `Unsolved`、结尾又给 `Solved`，属于歧义输出，不应直接取其中一个作为分数。query 3723 重放时得到可解析结果，说明评测输出也存在波动。详见旧实验目录的 `invalid_action_diagnosis.json` 和 `judge_format_diagnosis.json`；保留原始记录。

### 本次修复

- 按真实 `user → assistant → user` 消息角色组织交互；后续上下文只保留实际执行的 Action 和参数及环境返回，不回灌思考内容。无效输出用明确的未执行标记代替，原文仍保留在审计记录中。
- 默认 `enable_thinking=False`，在现有 1024-token 动作输出预算内直接生成动作；训练及验证采用 temperature 0.7、top_p 0.8、top_k 20。模式开关依据 [Qwen3 官方说明](https://huggingface.co/Qwen/Qwen3-1.7B#switching-between-thinking-and-non-thinking-mode)。
- 训练和预处理共享单动作规则，明确要求 JSON 后停止。vLLM 在 Observation、Tool response、Previous action、User query 边界停止；由代码常量设置真实换行，避免 Hydra 将 `\\n` 保留为字面字符串。
- 环境使用 vLLM 已按字符串停止的完成文本执行动作，同时保存原始 token 解码文本、结束原因、停止标记和原始 token 数。停止字符串可能还在 token ID 中，不能只重新解码 token ID 后执行。
- 对没有 EOS 的字符串停止输出，响应 attention/loss mask 按真实生成长度屏蔽 padding，避免将补齐 token 纳入学习。回归测试覆盖这一训练正确性问题。
- 确认生成器因长度限制停止而动作不完整时，明确反馈输出预算不足，要求下一轮使用简短、完整的 JSON；不把不完整动作当有效工具调用。
- 评分格式错误最多重试一次，保留两次原始输出与判定；最终仍歧义或格式不合法时保持 error，不强行转换成成功或失败。
- 真实调用发现 `autocomplete_cities.json` 中 Sacramento 的重复规范化键冲突会使整个 API 缓存加载失败，包括无关的 Memphis 查询。现只隔离冲突键，其他请求正常命中或转到模拟服务；真实 Memphis 请求已成功回退到 MirrorAPI-Cache。

### 实际模型检查与重新启动

在同一原始 Qwen3-1.7B 上选取旧验证集前 24 条任务，比较旧配置首轮与修复后的首轮，再运行修复后的最多 12 轮交互。该检查同时改变了模式、格式提示和停止策略，是整体协议检查，不是逐项消融。

| 项目 | 结果 |
| --- | ---: |
| 旧配置首轮有效动作 | 0/24 |
| 修复后首轮有效动作 | 18/24（75%） |
| 首轮触发长度停止 | 旧配置 5/24；修复后 0/24 |
| 修复后全程有效动作 | 103/123（83.74%） |
| 提交最终答案 | 18/24 |
| 最终评分异常 | 0 |
| MirrorAPI 判成功 | 1/24（4.17%） |
| 工具调用 | 82 |
| 磁盘缓存／模拟服务／执行异常 | 26／52／4 |

检查的 4 次工具异常中，2 次来自上述缓存文件的全局冲突，随后已修复并验证；另外 2 次为工具模拟回复截断，继续按工具执行异常记录。一个长列表最终答案反复达到长度限制，促使增加上述针对性截断反馈。该表是修复过程中这次固定检查的原始结果，没有回写成后续代码的成绩。

[检查摘要](../../outputs/toolbench_protocol_probe_20261001/summary.json)、[逐题检查结果](../../outputs/toolbench_protocol_probe_20261001/results.jsonl)、[可复用检查脚本](../../scripts/probe_toolbench_protocol.py)。

最终 **79 项 unittest** 通过，覆盖上下文角色、无效思考隔离、停止文本、无 EOS 的 padding mask、缓存冲突隔离和评分重试；启动命令经完整 Hydra 组合检查通过。

新版数据单独写入 `data/toolbench_stable_processed_v2`，训练 2048 条、验证 128 条，训练／验证标准化 query 交集仍为 0。共享协议增加提示词后重新做长度筛选，验证有 125/128 条与前版相同；3 条超长任务被后续合格任务替代，因此两个完整验证集合不能直接当作严格配对比较。新版初始输入最长为训练 3464 token、验证 2655 token，使用非思考聊天模板计数。

13:31 已在 GPU 1、2、5 从原始权重全新启动，保持 batch/minibatch 12、group 8、4096/1024 输入输出预算及每 10 步验证保存。旧错误输入实验已停止并保留结果。完整 128 条初始验证结果及评分校准见下一节。

- [本次启动记录](../../outputs/grpo_stabletoolbench_gpu125_protocolfix_20261001_133141/launch.json)
- [本次日志](../../outputs/grpo_stabletoolbench_gpu125_protocolfix_20261001_133141/train.log)
- 检查点：`/home/zhengbaowei/model/ckpt/grpo_qwen3_1.7b_stabletoolbench_gpu125_fresh_20261001_133141`


## 完整初始验证与 MirrorAPI 评分校准（13:47）

多轮协议修复版于约 13:47 完成 128 条初始验证。此时仍使用旧 `fac_prompt` 评分，得到 **9/128（7.03%）**，93/128（72.66%）提交答案；有效动作 **512/542（94.46%）**，11/542（2.03%）触发生成长度上限。工具调用恢复到 389 次，其中磁盘缓存 110 次、MirrorAPI-Cache 237 次、异常 42 次。初始验证之后已停止该实验，以便修正评分后从原始模型重新启动。

### 评分器存在独立于策略模型的偏差

对 8 组明确的完整／缺项答案做控制检查，原官方长 FAC 提示配合 MirrorAPI 仅判对 **11/16**。可复现的问题包括：首行 `Solved` 与后续解释中的 `Unsolved` 自相矛盾；声称答案缺少实际上已经提供的标题、链接、数值；声称缺项答案已提供该信息。对真实样本也观察到这些问题，因此原始 0 分和 24 条检查的 4.17% 都不能直接解释为策略模型真实能力。

仍按用户要求使用 **MirrorAPI**，新增 `fac_evidence`：按显式请求逐项核对证据，先写简短理由，再输出唯一的最终状态。保持完整性判据；拒答、缺项、要求用户提供答案仍判失败。官方 FAC 提示原文及其 SHA256 检查保留，原模式可选。默认评分类别明确为自定义 MirrorAPI 完整性评分，不宣称是专用 `stabletoolbench/Evaluator` 的官方指标。

新提示探索检查为开发样本 16/16、独立留出样本 15/16；最终生产适配器复测为 **开发 15/16、留出 15/16，合计 30/32（93.75%）**，0 格式异常。相同温度与种子下仍有判定变化，故不能保证完全确定。两条剩余误判均把完整答案判为缺项，全部保留，未修改预期标签来提高结果。这个很小的控制集不足以认证真实任务的评分准确率，尤其不保证事实正确性。

证据：[原提示校准](../../outputs/toolbench_judge_calibration_20261001.json)、[生产适配器复测](../../outputs/toolbench_judge_calibrated_adapter_20261001.json)、[可复用校准脚本](../../scripts/calibrate_toolbench_judge.py)、[固定控制样本](../../tests/fixtures/toolbench_judge_calibration.json)。

### 相同轨迹重评：34 成功、92 失败、2 异常

保持这次 128 条任务的模型输出、工具反馈及最终答案完全不变，只换评分提示重新判定：

| 项目 | 旧 `fac_prompt` | 新 `fac_evidence` |
| --- | ---: | ---: |
| 成功 | 9 | 34 |
| 失败，含 35 条未提交答案的协议失败 | 119 | 92 |
| 评分错误 | 0 | 2 |
| 有确定结论的覆盖率 | 128/128 | 126/128（98.44%） |
| 已评分子集成功率 | 9/128（7.03%） | **34/126（26.98%）** |

新评分的两条异常为 query 459、9346：两次响应均未遵守理由在先、状态在后的输出协议，其中 query 459 两次给出的结论也不一致。保留 error 与空分数，不放宽解析来强行补齐。已确认成功数占全体 34/128（26.56%），但这只是当前已确认成功的占比，**不是完整集合的最终成功率**。

这个对照只衡量评分方式的影响，不是训练收益。评分依据是任务覆盖／完整性，也未校验所有工具模拟内容的事实真实性。[重评摘要](../../outputs/toolbench_initial_regraded_20261001/summary.json)和[逐题新旧判定](../../outputs/toolbench_initial_regraded_20261001/results.jsonl)保留原判定，原始验证文件未覆盖。

### 工具异常与后续修复

42 次异常包括：24 次模拟回复缺失或截断、9 次请求本身命中冲突缓存键、6 次尚未发出 HTTP 请求便在本地并发槽排队超时、3 次模拟回复 JSON 非法。冲突缓存键仍按错误处理，避免任意选择一个不确定响应；不影响该 API 其他键。环境执行线程数现与客户端并发上限一致，让多余任务在执行器中排队，避免本地排队误报成工具服务超时。工具 HTTP 超时及非法响应仍显式记录。


对其中 8 条截断调用重新执行，将工具模拟输出预算从 2048 增至 4096、HTTP 超时增至 120 秒，**恢复 0/8**：7 条仍缺失／截断，1 条返回非法 JSON。涉及长列表、产品详情及二维码 base64；没有证据表明单纯增加预算能解决，因此未将这组参数投入训练。原始缓存及模拟服务回复继续按严格协议处理，不拼接或伪造完整结果。[失败重放记录](../../outputs/toolbench_simulator_4096_probe_20261001.json)。

本轮最终 **83 项 ToolBench unittest 通过**，新增覆盖评分理由／状态顺序、拒绝缺失或多重状态、格式重试、独立文本提示渲染兼容性；保留原官方提示哈希检查。校准工具、固定控制样本及全部代码修复纳入 Git，实验原始数据保留在本地输出目录。
