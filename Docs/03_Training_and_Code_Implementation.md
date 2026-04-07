# 混合训练与工程实现管线 (Training & Implementation Pipeline)

本规范详细记录了如何针对如此多模态、混合交叉损失（离散+连续扩散）、含有在线微调（DPO）特性的巨型架构系统，进行稳定、高效的代码实现。其核心引擎代码位于 `training/new_train.py` 之中。

## 1. DPO 混合数据流水环设计 (Data Loop Architecture)

因为兼顾基础生成能力与直接偏好引导，整个核心生命循环，在一个单一的 Epoch 里囊括了三个重型操作流水：

1. **基础去噪期 (Standard Diffusion Diffusion Stage)**：
   - 载入默认的 `DataLoader`。利用全局庞大的背景知识（例如全量的天然产物 NPs）前向传播加噪，并算取 MSE 与 BCE。
2. **在线抽样期 (Online Evaluation Sampling Stage)**：
   - 通过配置 `OnlineSamplingCallback`，在每一轮特定时期（通过 `dpo_sampling_every_n_epochs` 等频率调控）利用当前未被覆盖的权重生成样本，运用 `ShepherdScorer` 并行打分，存入到特定的内存缓存中组建 Preference Pairs。
3. **混合重载回推 (Mixed DPO Backprop Stage)**：
   - 使用工程函数 `create_mixed_dataloader` 创建**“混合数据集”**。按照规定的批次配比（如 7 份标准库 vs 3份偏好对），同时把两类任务推进去送显存以防灾难性遗忘。

## 2. Dataloader 与 `HeteroDataset` 的重构兼容

工程的重心在于数据拼装层面（$x1$ 的图、$x2$ 的等离子点云、$x4$ 的具备定向特征的药效团）：
- **参数控制多维分离**：通过独立定义的 `compute_x1`、`compute_x2`...可以独立开关各类表征的训练强度。且它们的 $T$ 级噪声策略也彼此分离。
- **边界统计缓存 (Marginal Distributed Cache)**：
  这非常重要：离散加噪的马尔可夫演变需要全局类别概率。在加载全量文件前，我们通过并行使用 `multiprocessing` 执行 `compute_and_cache_marginals()` 来构建所有可能原子的独立统计频次，确保其先验正确。
- **PyG DataLoader 兼容补丁**：
  在原始项目尝试融合 DPO 操作时，引发了重大由于 `node_types` 检查失敏（`hasattr` false）导致的一系列 Bug。在当前版本，放弃了可能导致 collate 覆写损坏的普通 DataLoader 传参规则，严格启用了安全模式进行跨域调度。

## 3. 分布式性能与内存高可用配置

大规模图扩散任务极为吃紧内存 IO 与进程队列通信，特在代码开头执行了如下工程保障：

### 3.1 Unix 文件句柄解除
由于 `HeteroDataset` 存在大规模图结构的零散读写和 Pytorch Worker 分时通讯，为了防范 "too many open files" 灾难：
```python
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
```

### 3.2 显存并行保障
- 采用硬件感知的 `file_system` 为共享策略，切断内存页污染。
- 引入 `Medium` 等级的深度学习核心矩阵乘，开启 `cudnn.benchmark = True` 提升并行乘加效率。
- 在构建 `Trainer` 时：如果开启 DPO，强制打开 `find_unused_parameters=True`，让在进行混合 DDP 模型训练时，即使有些层参与纯纯的生成预测但是并不产生对应的 Reference 损失也能安全通过后向回溯。

## 4. 关键参数设置字典说明

要对微调表现进行深度探索，必须针对 `parameters/` 内部文件做细致控制：
- `beta_dpo`：KL惩罚大小。如果产生模型只出一种长相一样的骨架（模式坍塌），证明对原连续去噪权重的依赖断裂了，需要**降低此值**。
- `dpo_ramp_up_epochs`：平滑引入系数。用来防止强行在初级 Epoch （此时主模型还在找方向无暇顾及细分奖励）加入 DPO 导致梯度冲突。
- `ema_alpha`：参考模型追赶步伐。若为0则参考模型一直是第10世代的老骨架。0.01 的缓慢游移让它处于稳定的半衰期拖影状态，可以长期跟随。
