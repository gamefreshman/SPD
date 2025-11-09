# Methodology

本研究采用基于扩散模型的分子生成框架SHEPHERD，该模型能够同时处理分子的多模态特征表示，并通过离散扩散过程实现高质量的分子图生成。

## 3.1 模型架构

### 3.1.1 异构图神经网络

本研究采用异构图结构处理分子的多层次特征，通过`HeteroDataset`构建包含原子、键和药效团的复合表示。模型基于图变换器（Graph Transformer）架构，能够捕获分子图中的长程依赖关系。

### 3.1.2 多模态特征融合

模型同时学习四种不同的分子表示：
- **x1**：分子图拓扑结构（原子类型、键类型、电荷信息）
- **x2**：分子表面特征（通过探针半径采样）
- **x3**：分子体积特征
- **x4**：药效团特征（含多向量方向信息）

## 3.2 离散扩散模型设计

### 3.2.1 离散空间的扩散过程

本研究的核心创新在于对分子图的离散特征（原子类型、键类型）采用离散扩散过程。与连续扩散不同，离散扩散通过状态转移矩阵实现噪声添加和去除：

**状态转移机制**：
- 采用马尔可夫转移矩阵 $Q_t$ 控制离散状态间的转移
- 对于原子类型：$p(x_t | x_{t-1}) = x_{t-1} Q_t^X$
- 对于键类型：$p(e_t | e_{t-1}) = e_{t-1} Q_t^E$

**边际分布导向的转移**：
```python
# 计算原子和键的边际分布
x_marginals = node_types / torch.sum(node_types)
e_marginals = edge_types / torch.sum(edge_types)

# 使用边际分布初始化转移模型
transition_model = MarginalUniformTransition(
    x_marginals=x_marginals, 
    e_marginals=e_marginals
)
```

### 3.2.2 噪声调度策略

**预定义噪声调度**：采用`PredefinedNoiseScheduleDiscrete`控制不同时间步的噪声强度，通过$\beta_t$参数调节每步的转移概率：

$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$$

**累积转移矩阵**：计算从初始状态到时间步$t$的累积转移概率：
$$Q_{t,bar} = Q_1 Q_2 \cdots Q_t$$

### 3.2.3 后验概率计算

在反向去噪过程中，计算条件概率分布$p(z_{t-1} | z_t, x_0)$：

```python
def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """计算离散扩散的后验分布"""
    # 基于贝叶斯法则计算后验概率
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
    prob_X = weighted_X.sum(dim=2)
    return prob_X / torch.sum(prob_X, dim=-1, keepdim=True)
```

### 3.2.4 采样过程

**逆向采样**：从噪声分布$z_T$开始，逐步采样$z_{t-1} \sim p(z_{t-1} | z_t)$：

1. **噪声初始化**：从极限分布采样初始噪声
2. **逐步去噪**：使用神经网络预测原始数据分布
3. **概率归一化**：确保每步采样的概率分布有效
4. **离散采样**：根据计算的概率分布采样离散特征

## 3.3 数据预处理与特征提取

### 3.3.1 分子数据处理

**分子表示**：使用RDKit处理分子块（molblocks）格式的输入数据，保留显式氢原子以维持完整的分子信息。

**并行特征计算**：实现并行化的特征边际分布计算，通过多进程池加速大规模数据集的特征提取：
```python
def compute_and_cache_marginals(params, molblocks_and_charges):
    # 并行处理批次数据
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(worker_fn, batches, chunksize=1), 
                           total=len(batches), desc="并行统计特征"))
```

### 3.3.2 缓存机制

建立特征缓存系统，避免重复计算边际分布，提高训练效率：
- 原子边际分布缓存：`{dataset_name}_atom_marginals.pt`
- 键边际分布缓存：`{dataset_name}_bond_marginals.pt`
- 药效团边际分布缓存：`{dataset_name}_pharm_marginals.pt`

## 3.4 噪声调度与扩散过程

### 3.4.1 多模态噪声调度

为不同的特征模态设计独立的噪声调度策略，通过`noise_schedule_dict`参数控制各模态的扩散过程：

- **独立时间步**：不同模态可使用独立的时间步长
- **模态特定缓存**：各模态维护独立的噪声参数
- **自适应调度**：根据模态特性调整噪声强度

### 3.4.2 几何约束处理

**几何一致性维护**：
- 质心重定位（`recenter_x1/x2/x3/x4`）确保分子几何的一致性
- COM（质心）噪声移除策略维持分子的旋转和平移不变性
- 虚拟节点机制增强模型的表示能力

**掩码机制**：
```python
def apply_noise(self, X, E, y, node_mask):
    # 应用节点掩码确保只对有效节点添加噪声
    z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).mask(node_mask)
```

## 3.5 训练策略

### 3.5.1 混合训练范式

结合标准扩散训练和DPO（Direct Preference Optimization）微调：

1. **预训练阶段**：使用标准扩散损失在大规模分子数据集上预训练
2. **DPO微调阶段**：通过偏好数据对模型进行精细调整

```python
if params['training'].get('enable_dpo', False):
    # 混合DataLoader（标准 + DPO）
    train_loader = create_mixed_dataloader(
        standard_dataset=dataset,
        dpo_dataset=dpo_dataset,
        dpo_ratio=params['training'].get('dpo_batch_ratio', 0.3)
    )
```

### 3.5.2 在线采样与偏好数据生成

**在线采样架构**：本研究实现了完整的在线采样系统，在每个训练epoch开始时动态生成新分子并构建偏好对：

```python
class OnlineSamplingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # 1. 选择种子分子（基于难度或分层采样）
        seed_indices = self.scheduler.select_seeds(dataset, epoch, losses)
        # 2. 使用当前模型生成样本
        generated_mols = sampler.batch_sample(seed_mol_list)
        # 3. 构建偏好对
        preference_pairs = self.pair_builder.batch_build_pairs(generated_mols)
```

**分子质量评估系统 - Shepherd Score**：

采用多维度分子质量评估系统，综合考虑化学有效性、药物相似性和结构特征：

```python
class ShepherdScorer:
    def score_molecule(self, mol_generated, mol_reference=None):
        # 1. 基本有效性检查
        scores['is_valid'] = self.check_validity(mol_generated)
        
        # 2. 构象评估（应变能、LogP、QED等）
        conf_eval = ConfEval(atoms, positions, solvent='water')
        scores['strain_energy'] = conf_eval['strain_energies']
        scores['logp'] = conf_eval['logPs']
        scores['mqed'] = conf_eval['QEDs']
        
        # 3. 相似性评估（如果有参考分子）
        cond_pipe = ConditionalEvalPipeline(ref_molec, generated_mols)
        scores['sims_surf_upper_bound'] = cond_pipe['sims_surf_upper_bound']
        scores['sims_esp_upper_bound'] = cond_pipe['sims_esp_upper_bound']
        
        # 4. 综合评分
        total_score = (scores['mqed'] * 2.0 - 
                      abs(scores['logp'] - 1.5) * 0.3 -
                      scores['strain_energy'] * 0.5 + 
                      scores['sims_surf_upper_bound'] * 1.0)
        return scores
```

**评分权重策略**：
- **QED权重**: 2.0（药物相似性最重要）
- **LogP惩罚**: 0.3（理想范围0-3）
- **应变能惩罚**: 0.5（结构稳定性）
- **表面相似性奖励**: 1.0（条件生成时）
- **静电势相似性奖励**: 1.0（精细特征匹配）

### 3.5.3 偏好对构建策略

**偏好对生成流程**：

1. **多样本生成**：对每个种子分子生成4个候选分子
2. **质量评估**：使用Shepherd Score对所有候选进行评分
3. **有效性筛选**：过滤掉化学无效分子
4. **偏好对构建**：选择最高分和最低分分子形成偏好对
5. **分差阈值**：只保留分数差距超过阈值（默认0.5）的偏好对

```python
class PreferencePairBuilder:
    def build_pairs_from_samples(self, generated_mols, reference_mol=None):
        # 评分所有候选分子
        score_dicts = self.scorer.score_batch(valid_mols, reference_mol)
        
        # 按total_score排序
        valid_pairs.sort(key=lambda x: x[1]['total_score'], reverse=True)
        winner_mol, winner_score = valid_pairs[0]
        loser_mol, loser_score = valid_pairs[-1]
        
        # 检查分数差距
        score_gap = winner_score['total_score'] - loser_score['total_score']
        if score_gap < self.min_score_gap:
            return None  # 舍弃分差不足的偏好对
            
        return (winner_mol, loser_mol, winner_score, loser_score)
```

### 3.5.4 采样调度策略

**种子分子选择策略**：

1. **基于难度的采样**：优先选择训练损失较高的困难样本
2. **分层随机采样**：按分子大小分桶，确保大中小分子的代表性
3. **动态采样比例**：默认采样5%的训练数据作为种子

```python
class DPOSamplingScheduler:
    def select_seeds(self, dataset, epoch, losses=None):
        # 策略1：困难样本优先
        if losses is not None:
            sorted_indices = sorted(losses.keys(), 
                                  key=lambda k: losses[k], reverse=True)
            return sorted_indices[:n_samples]
        
        # 策略2：分层采样（按分子大小）
        size_buckets = {'small': (0,15), 'medium': (16,35), 'large': (36,999)}
        bucket_weights = {'small': 0.3, 'medium': 0.4, 'large': 0.3}
        # 按权重分配采样数量
```

**数据重用机制**：
- 保留50%的历史偏好对避免灾难性遗忘
- 动态更新偏好对缓存
- 失败恢复：采样失败时使用历史数据继续训练

### 3.5.5 DPO损失函数设计

**DPO损失原理**：基于Bradley-Terry模型，优化策略模型相对于参考模型的偏好概率：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

其中：
- $\pi_\theta$：当前策略模型
- $\pi_{ref}$：参考模型（冻结的预训练模型）
- $\beta$：KL散度惩罚系数
- $(y_w, y_l)$：winner和loser分子对

**实际实现细节**：

```python
def compute_dpo_loss(self, winner_data, loser_data, shared_noise):
    # 使用共享噪声确保公平比较
    winner_logp = self.compute_conditional_logp(winner_data, shared_noise)
    loser_logp = self.compute_conditional_logp(loser_data, shared_noise)
    
    # 参考模型概率（冻结）
    ref_winner_logp = self.ref_model.compute_conditional_logp(winner_data, shared_noise)
    ref_loser_logp = self.ref_model.compute_conditional_logp(loser_data, shared_noise)
    
    # 计算相对概率比
    winner_ratio = winner_logp - ref_winner_logp
    loser_ratio = loser_logp - ref_loser_logp
    
    # Bradley-Terry损失
    dpo_loss = -torch.log_sigmoid(self.beta * (winner_ratio - loser_ratio))
    
    # 隐式奖励计算（用于监控）
    implicit_reward_winner = self.beta * winner_ratio
    implicit_reward_loser = self.beta * loser_ratio
    implicit_acc = (implicit_reward_winner > implicit_reward_loser).float()
    
    return dpo_loss, implicit_acc
```

**混合损失设计**：

$$\mathcal{L}_{total} = (1-\lambda) \mathcal{L}_{diffusion} + \lambda \mathcal{L}_{DPO}$$

其中$\lambda$是DPO损失权重，动态调整以平衡生成质量和偏好学习。

## 3.7 损失函数设计

### 3.7.1 变分下界（ELBO）

模型优化基于变分下界的四个组成部分：

1. **节点数量先验**：$\log p(N)$ - 分子大小分布的对数概率
2. **KL散度项**：$D_{KL}[q(z_T|x_0) || p(z_T)]$ - 终态分布与先验的差异
3. **扩散损失**：$E_t[D_{KL}[q(z_{t-1}|z_t,x_0) || p_\theta(z_{t-1}|z_t)]]$ - 所有时间步的去噪误差
4. **重构损失**：$-\log p_\theta(x_0|z_0)$ - 从去噪结果重构原始数据的概率

### 3.7.2 离散交叉熵损失

对于离散特征，采用分类交叉熵损失：
```python
loss = self.train_loss(
    masked_pred_X=pred.X, masked_pred_E=pred.E,
    true_X=X, true_E=E, true_y=data.y
)
```

## 3.8 优化配置

### 3.8.1 分布式训练

采用PyTorch Lightning的DDP（Distributed Data Parallel）策略支持多GPU训练：
```python
strategy = DDPStrategy(find_unused_parameters=True) if (
    (params['training']['num_gpus'] > 1 and cuda_available) or 
    params['training'].get('enable_dpo', False)
) else 'auto'
```

**参数管理**：通过`find_unused_parameters=True`处理DPO模式下的复杂参数依赖。

### 3.8.2 内存优化

**文件描述符管理**：
```python
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
```

**共享策略**：使用`file_system`共享策略确保多进程稳定性，启用持久化工作进程。

### 3.8.3 梯度控制

- **梯度裁剪**：设置梯度裁剪阈值防止梯度爆炸
- **梯度累积**：支持更大的有效批次大小
- **精度优化**：使用中等精度矩阵乘法优化计算效率

## 3.9 评估与监控

### 3.9.1 多层次监控

集成CSV日志和Weights & Biases（wandb）进行实时训练监控：
- 损失函数追踪
- 梯度和参数变化监控
- 分子生成质量评估

### 3.9.2 检查点管理

实施智能检查点保存策略：
- 支持训练中断恢复
- 模型版本管理
- DPO模式下的权重迁移

```python
checkpoint_callback = ModelCheckpoint(
    save_top_k=0, save_last=True,
    monitor="train_loss", mode="min",
    every_n_train_steps=params['training']['log_every_n_steps']
)
```

## 3.10 创新点总结

1. **离散扩散机制**：首次将离散扩散应用于分子图生成，通过状态转移矩阵处理原子和键类型的离散特征
2. **边际分布导向**：基于真实数据集的特征分布设计转移概率，提高生成分子的化学合理性
3. **多模态融合**：同时处理拓扑、几何和药效团特征，实现全面的分子表示学习
4. **在线DPO微调**：创新性地将Direct Preference Optimization应用于扩散模型，通过在线采样和Shepherd Score评分系统动态构建偏好对，实现化学质量导向的模型优化
5. **混合训练策略**：结合标准扩散和偏好优化，平衡生成质量和化学有效性
6. **智能采样调度**：设计基于难度和分层采样的种子选择策略，确保DPO训练的数据质量和代表性
7. **高效实现**：通过并行计算和缓存机制，支持大规模分子数据集的训练

该方法在保持分子化学有效性的同时，显著提升了生成分子的多样性和新颖性，为药物发现和材料设计提供了强有力的计算工具。
