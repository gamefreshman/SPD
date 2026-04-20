# DPO训练说明

## 1. 目的

这套 DPO 训练不是在替代原有的 diffusion 监督训练，而是在现有 `x1/x3/x4` 条件扩散框架上增加一层**偏好优化**，让模型在固定参考 NP 条件下，更倾向于生成评分更高的分子构象/候选分子。

这里的核心目标不是单纯提高重建 loss，而是优化下面这件事：

- 在同一个 reference NP 提供的条件 `x3/x4` 下，模型应该更偏好 `winner`，而不是 `loser`

从训练语义上，DPO 学的是：

- `p_theta(winner_x1 | ref_x3x4) > p_theta(loser_x1 | ref_x3x4)`

这也是这次实现里最重要的修正点：**winner 和 loser 必须共享同一个 reference condition，而不能各自带自己的 `x3/x4`。**

## 2. 整体结构

基于这次迁移历史，需要先区分两类文件：

- `SPD/training` 下的历史实验脚本与分析脚本
- 迁移到 Shepherd 风格后的稳定 DPO 实现

当前本地 `SPD/training` 目录里，和 DPO 直接相关的文件主要是：

- `training/DPO1_0_triSim.py`
- `training/new_train.py`
- `training/parameters/params_x1x3x4_dpo_finetune_nps.py`
- `training/visualize_dpo_metrics.py`
- `training/DpoAndVisual.md`

而这次迁移后更接近稳定实现的主链路，主要保存在：

- `.migration_shepherd/training/dpo_train.py`
- `src/shepherd/dpo_dataset.py`
- `src/shepherd/dpo_utils.py`
- `src/shepherd/mixed_dataloader.py`
- `src/shepherd/lightning_module.py`

下面的说明以“迁移后稳定实现”为主，同时补充 `training` 目录里仍然保留的历史脚本角色。

各模块职责如下。

### 2.1 `.migration_shepherd/training/dpo_train.py`

主入口，负责把整个 DPO 训练链路串起来：

- 读取参数
- 加载 reference NPs
- 构建 `reference_cache`
- 加载基础模型与 checkpoint
- 构建监督数据集 `HeteroDataset`
- 在线采样 preference pairs
- 构建 `DPODataset`
- 用 `MixedDataLoader` 将监督 batch 与 DPO batch 混合
- 配置 `Lightning` 的 callback、checkpoint、日志和训练器

它同时还负责 epoch 间的在线重采样，也就是：

- 训练若干 epoch
- 用当前模型重新生成样本
- 重新打分并更新 preference pairs
- 继续训练

补充说明：

- `training/DPO1_0_triSim.py` 是更早期的 DPO 训练实验脚本，历史上用过 `ConfEval` 和旧版评分链。
- `training/new_train.py` 是另一条训练脚本分支，保留了“标准扩散训练 + DPO 微调”的实验入口，但不是这次共享 reference condition 修复后的主实现说明对象。

### 2.2 `src/shepherd/dpo_dataset.py`

负责把 online sampling 产出的 preference pairs 转成模型实际可消费的 batch。

这里有两个关键职责：

- 为 winner / loser 分别构造 `x1`
- 为二者共享构造同一个 reference 的 `x3/x4`

因此它不再是“从 winner 自己重建一份 `x3/x4`，从 loser 自己再重建一份 `x3/x4`”，而是：

- `winner = winner_x1 + shared_ref_x3x4`
- `loser = loser_x1 + shared_ref_x3x4`

这是 DPO 语义成立的基础。

### 2.3 `src/shepherd/dpo_utils.py`

负责在线采样与偏好对构造，主要包括：

- `OnlineSampler`
- `MultiGPUOnlineSampler`
- `ShepherdScorer`
- `PreferencePairBuilder`

职责拆分如下：

- `Sampler`: 在 reference 条件下生成候选样本
- `Scorer`: 对候选样本算 surface / ESP / pharm 等相似度分数
- `PairBuilder`: 从一组带分数样本中构造 winner / loser 对

### 2.4 `src/shepherd/mixed_dataloader.py`

负责把两种不同来源的 batch 混合：

- 原始监督训练 batch
- DPO 偏好学习 batch

这样训练不是纯 DPO，而是**监督损失 + DPO 损失**的混合优化。

### 2.5 `src/shepherd/lightning_module.py`

真正计算 loss 的地方。

这里同时支持：

- 原来的 diffusion denoising / supervision loss
- 新增的 DPO loss

同时它还持有：

- `ref_model`
- DPO 权重 ramp-up 参数
- `beta_dpo`
- `dpo_max_weight`
- `dpo_optimize_x4`

当前稳定实现里，`dpo_optimize_x4` 应明确为 `False`，也就是：

- `x4` 只作为条件输入参与生成
- DPO 不直接对 `x4` 通道做偏好优化

## 3. 训练数据流

整体数据流可以概括成下面这条链：

1. 从 reference NP 列表中取一批参考分子
2. 在其 `x3/x4` 条件下在线采样若干候选分子
3. 对候选分子做 chemical validity 检查与相似度打分
4. 在同一个 reference 内部构造 winner / loser preference pairs
5. 用这些 pairs 构建 `DPODataset`
6. 用 `MixedDataLoader` 把监督 batch 和 DPO batch 混合
7. `LightningModule` 同时计算标准监督 loss 与 DPO loss
8. 训练若干 epoch 后再次采样，刷新 pair buffer

对应的核心思想是：

- reference 提供条件
- online sampling 提供候选
- scorer 提供偏好信号
- DPO 负责把偏好信号反向灌回模型

## 4. 关键实现细节

### 4.1 共享 reference condition

这是这套 DPO 里最重要的约束。

如果 winner / loser 各自使用自己的 `x3/x4`，训练就会退化成：

- `p(winner | winner_condition)` vs `p(loser | loser_condition)`

这不再是在比较“同一道题下两个候选答案”，而是变成了两道题的比较，DPO 语义会偏掉。

正确实现必须是：

- `winner_x1 + ref_x3/x4`
- `loser_x1 + ref_x3/x4`

因此 pair 数据结构里必须显式带 `reference_idx`，并由 `DPODataset` 通过 `reference_cache` 重建共享条件。

### 4.2 `reference_cache`

`reference_cache` 的职责是缓存 reference 相关的静态信息，而不是在每个 pair 里重复塞一整份 reference 数据。

这样做有两个原因：

- 节省 pair 存储体积
- 保证 winner / loser 使用的 reference condition 真正来自同一个缓存对象语义

通常 pair 只需要保存：

- `winner`
- `loser`
- `reference_idx`
- 可选的 winner / loser score

而 reference 的 molblock、partial charges、surface / pharm 条件等统一放在 `reference_cache` 中。

### 4.3 `x1` 与 `x3/x4` 的时间步关系

winner 和 loser 在同一个 pair 内，必须共享：

- reference `x3/x4`
- 同一个 DPO 比较时间步 `t_idx`

但这不意味着 `x1` 的噪声张量要完全复用。

更准确的约束是：

- `x1` 分别来自 winner 和 loser 自身
- 比较时采用同一个时间步
- `x3/x4` 来自同一个 reference

实现上更稳的方式不是“同 seed 各跑一遍”，而是：

- reference condition 只构造一次
- 然后安全地拷贝给 winner / loser

### 4.4 为什么 `dpo_optimize_x4=False`

当前实现中，`x4` 是条件，不是偏好目标。

如果把 `x4` 也纳入 DPO channel，会有两个问题：

- 训练目标变复杂，偏离“固定条件下比较 `x1`”这个核心定义
- winner / loser 已经共享 reference `x4`，对 `x4` 再做偏好比较本身就不自然

因此稳定实现中建议保持：

- `dpo_optimize_x4 = False`

也就是：

- `x4` 继续作为条件
- DPO 只比较 `x1` 相关的生成偏好

### 4.5 混合训练，而不是纯 DPO

训练时不会只喂 DPO 数据，而是把两类 batch 混合：

- 标准监督 batch，保证基础生成能力不丢
- DPO batch，提供偏好优化信号

这一点非常重要。否则模型容易出现：

- 过快偏离原始扩散分布
- 有效性下降
- 训练不稳定

参数上通常通过以下项控制混合比例和强度：

- `real_data_ratio`
- `beta_dpo`
- `dpo_max_weight`
- `dpo_ramp_up_epochs`

其中 `dpo_ramp_up_epochs` 的作用是：

- 不让 DPO 一开始就给太大权重
- 先保住基础生成，再逐步加重偏好优化

### 4.6 在线采样与 buffer 刷新

DPO 训练不是一次性离线准备好所有 winner / loser pairs，而是周期性地在线重采样。

原因很直接：

- 模型在变
- 偏好对也应该随当前模型分布更新

因此 `dpo_train.py` 里会有：

- 初始采样
- 每隔若干 epoch 的重采样
- pair buffer 刷新

常用控制项包括：

- `dpo_sampling_every_n_epochs`
- `num_initial_references`
- `num_references_per_update`
- `num_samples_per_molecule`
- `inference_sub_batch_size`

### 4.7 多 GPU 采样，但训练仍是单 GPU

后续实现中加入了 `MultiGPUOnlineSampler`，目的是并行化在线采样，缓解 sampling 阶段速度瓶颈。

这里要区分两件事：

- **训练多卡**：当前没有做成 DDP 训练
- **采样多卡**：已经支持

也就是说，目前的稳定策略是：

- 训练主进程仍单卡
- 在线 sampling 可分发到多张 GPU 并行执行

对应配置一般是：

- `training.num_gpus = 1`
- `dpo.sampling_gpu_ids = [...]`

### 4.8 多 GPU 采样里的两个关键约束

### 1. 只复制推理必需对象

多卡采样时不能简单 `deepcopy(LightningModule)`，因为那会把 `ref_model` 等整套状态一起复制，显存成本过高。

正确做法是：

- 只复制采样需要的 `model`
- 包一层轻量 `_InferenceWrapper`
- 显式提供 `device`、`params` 等推理必需属性

### 2. `group_id` 语义必须保持稳定

pair 构造时，`group_id` 代表的是“同一个 reference 下同一组候选样本”的比较范围。

即使采样任务被分发到多个 GPU，并行返回的先后顺序也不能改变这个语义。否则会出现：

- 不同 sub-batch 的样本被错误混组
- winner / loser 比较范围改变
- preference pair 语义漂移

因此 `group_id` 必须在 task 创建阶段就稳定分配，而不能按 future 返回顺序现编。

### 4.9 `pair_key` 必须带 `reference_idx`

如果 pair 去重只看：

- `winner_signature`
- `loser_signature`

而不看 `reference_idx`，就会把“同一对分子在不同 reference 条件下的比较”错误去重掉。

这是不对的。

因为：

- 同样的 winner / loser
- 在不同 reference 条件下
- 属于不同训练样本

所以去重键必须带 reference identity。

## 5. 打分与相似度

DPO 的 preference signal 来自 `ShepherdScorer`。它一般会综合以下几类相似度：

- surface similarity
- electrostatic similarity
- pharmacophore similarity

再按设定权重合成总分。

在偏好构造中通常会保留：

- `winner_scores`
- `loser_scores`
- `score_gap`

后续训练和分析时，这些指标可以帮助判断：

- 偏好对是否足够清晰
- 是否存在大量弱 pair
- 哪个相似度通道贡献更大

## 6. 可视化与产物

训练过程中一个很重要的中间产物是：

- `dpo_round_metrics.json`

它会记录每轮在线采样/重采样后的核心指标，例如：

- `num_pairs`
- `avg_score`
- `validity_rate`
- `pairable_count`
- `implicit_acc`
- `score_gap`
- `sampling_error`

对应的可视化脚本是：

- `training/visualize_dpo_metrics.py`

典型用法：

```bash
cd training
python visualize_dpo_metrics.py jobs/x1x3x4_dpo_finetune_nps/dpo_round_metrics.json --output jobs/x1x3x4_dpo_finetune_nps/dpo_round_metrics.png
```

这个脚本主要用于检查三类问题：

- 训练是否稳定产生 pair
- 有效率是否持续过低
- winner / loser 的 gap 是否在改善

## 7. 常见参数及含义

下面这些参数是实际调参时最重要的。

### 7.1 DPO 权重相关

- `beta_dpo`
  - DPO 偏好强度
- `dpo_max_weight`
  - DPO loss 最大混合权重
- `dpo_ramp_up_epochs`
  - DPO 权重 warm-up 周期

### 7.2 采样相关

- `num_samples_per_molecule`
  - 每个 reference 生成多少候选
- `inference_sub_batch_size`
  - 每次推理采样的子 batch 大小
- `sampling_gpu_ids`
  - 参与在线采样的 GPU 列表

### 7.3 重采样频率相关

- `dpo_sampling_every_n_epochs`
  - 每隔多少 epoch 重新采样一次
- `num_initial_references`
  - 初始采样的 reference 数
- `num_references_per_update`
  - 每轮更新时采样多少个 reference

### 7.4 训练保护相关

- `buffer_gate_*`
  - 用于控制 pair buffer 是否达标
- `protect_stop_*`
  - 用于在训练质量明显不达标时提前停下或保护性跳过

这些保护项的目标不是“追求绝对干净”，而是避免：

- 连续多轮没有有效 pair
- 低质量 pair 持续灌入训练
- 训练在坏分布上越跑越偏

## 8. 已知问题与经验

### 8.1 空 round 不是异常，但连续空 round 是问题

在线采样时出现单次 `num_pairs = 0` 并不一定代表训练坏掉，尤其在生成样本化学有效性波动较大时。

真正要警惕的是：

- 连续多轮 empty round
- `validity_rate` 长期过低
- `sampling_error` 非空

### 8.2 RDKit / 化学有效性警告会大量出现

在实际运行中，常见问题包括：

- `Explicit valence ... greater than permitted`
- `Can't kekulize mol`
- 缺显式氢

这些不一定会立刻让训练崩溃，但如果比例过高，会直接导致：

- `num_valid` 下降
- `num_scoreable` 下降
- pair 数不足

因此实际运行时比 `loss_dpo` 更值得盯的是：

- `validity_rate`
- `num_pairs`
- `score_gap`

### 8.3 初始采样失败通常不是训练器问题，而是上游链路问题

如果日志里出现：

- `Initial DPO sampling produced no preference pairs`

优先检查的不是 `LightningTrainer`，而是：

- 样本是否生成成功
- validity check 是否全部失败
- scorer 是否因为分子重建问题无法打分
- pair builder 是否由于 gap 阈值过高或 grouping 过窄筛空了样本

### 8.4 scorer 与远端环境兼容性要谨慎

实践中，最容易出问题的往往不是训练框架本身，而是 scorer 依赖链，比如：

- `ConfEval`
- `convert_data`
- RDKit 分子重建逻辑

如果本地和远端环境中的 `score` 实现版本不一致，常见现象是：

- 采样能跑
- 但全部 sample 无法有效评分
- 最后表现为 `no preference pairs`

因此部署时要优先确认：

- `score` 代码来源是否一致
- `shepherd_score`/`ConfEval` 是否是当前训练逻辑依赖的正确版本

## 9. 推荐的监控重点

如果只看少量关键指标，建议优先盯下面这些。

### 第一优先级

- `validity_rate`
- `num_pairs`
- `sampling_error`

这三项决定训练是不是在持续获得有效偏好信号。

### 第二优先级

- `avg_score`
- `score_gap`
- winner / loser 的 `surf/esp/pharm` gap

这几项决定偏好信号的质量是否在变强。

### 第三优先级

- `loss_dpo`
- `implicit_acc`

这些指标有价值，但通常波动更大，也更容易受 pair 数不足影响，不能孤立解读。

## 10. 一句话总结

这套 DPO 训练的本质是：

- 用 reference NP 提供共享条件
- 用在线采样生成候选
- 用外部相似度 scorer 给出偏好信号
- 用 DPO 将这种偏好反向灌回 diffusion 模型

真正决定它能不能跑稳、跑出效果的，不只是 loss 是否下降，而是下面这几件事是否同时成立：

- winner / loser 是否共享同一个 reference condition
- scorer 是否稳定给出有效分数
- pair buffer 是否持续有料
- 监督训练与 DPO 的混合比例是否合适
- 采样阶段是否足够快且不过度 OOM
