### 一、整体架构


- [ ] 训练模式（配置一个参数）
	- [ ] 一个 epoc 中 只进行 标准训练
	- [ ] 一个 epoc 中 两种训练都进行
- [ ] multiple_train_dataloader 

1.  **加载固定的参考模型 (`ref_model`)**
    *   在`LightningModule`的`setup`或`__init__`中，加载你之前训练好的模型权重到`self.ref_model`。
    *   **立即冻结**：`self.ref_model.eval()` 和 `for param in self.ref_model.parameters(): param.requires_grad = False`。

2.  **实现Shepherd Score 
	- [ ] 确定重要的参数指标
		- [ ] rmsd
		- [ ] strain_energy
		- [ ] LogP
		- [ ] MQED
		- [ ] is valid
		- [ ] sims_surf_upper_bound
		- [ ] sims_esp_upper_bound
	- [ ] 从 evaluation/main.ipynb 中，了解如何使用已有的评估函数
	- [ ] 只保留四个中，差距分数最大的两个
	- [ ] 如果分差太低，就不加入到偏好对列表

3.  **实现在线采样逻辑**
    *   参考 evaluation/main.ipynb 的采样+评估逻辑，写一个函数，输入 一个完整的分子 和当前模型`model`，输出`4`个生成的分子。
    *   调用Shepherd Score，构建偏好对列表。
    * 保留50%上一轮偏好对 + 50%新生成，提高数据利用率
    * 保留样本 implicit_acc < 0.6 的偏好对：模型还没学好，保留
    * 保留损失仍然高的偏好对

4.  **实现DPO损失函数**
    *   这是核心代码。严格按照你的方案中的公式编写，同时处理连续和离散变量。
    *   注意**共享噪声和时间步`t`是关键。
    * 连续变量: noise matching 
    * 离散变量: x0 matching

**DPO权重计算公式**

def get_dpo_weight(epoch):
    if epoch < 5:
        return 0.1 * (epoch+1)
    else:
        return 0.5  # 稳定期
#### **计算损失**

**标准去噪损失**（用于混合训练）：
- 把winner样本当作普通训练样本
- 计算标准的去噪损失
- `loss_std = standard_loss(model, x1_w, x4_w)`

**DPO损失**（连续变量）：
```
model_loss_w = MSE(noise_w_pred, noise)
model_loss_l = MSE(noise_l_pred, noise)
model_diff = model_loss_w - model_loss_l

ref_loss_w = MSE(noise_w_ref, noise)
ref_loss_l = MSE(noise_l_ref, noise)
ref_diff = ref_loss_w - ref_loss_l

inside_term = -0.5 × beta_dpo × (model_diff - ref_diff)
loss_dpo_continuous = -log(sigmoid(inside_term))
```

**DPO损失**（离散变量）：
```
model_loss_w = CrossEntropy(logits_w, x_w_true)
model_loss_l = CrossEntropy(logits_l, x_l_true)
model_diff = model_loss_w - model_loss_l

ref_loss_w = CrossEntropy(logits_w_ref, x_w_true)
ref_loss_l = CrossEntropy(logits_l_ref, x_l_true)
ref_diff = ref_loss_w - ref_loss_l

inside_term = -0.5 × beta_dpo × (model_diff - ref_diff)
loss_dpo_discrete = -log(sigmoid(inside_term))
```

**总DPO损失**：
```
loss_dpo_total = loss_dpo_continuous + loss_dpo_discrete
```

**混合损失**：
```
loss_final = 0.7 × loss_std + 0.3 × loss_dpo_total
```


5.  **改造训练循环 (`training_step`)**
    *   这是最复杂的修改。你需要一种机制来处理两种数据：标准数据和偏好对数据。

6.  **实现DPO权重Ramp-up**
    *   在`LightningModule`中，根据`self.current_epoch`动态计算`dpo_weight`。
    *  预热策略

## 监控指标体系

### **1. 标准训练指标（继续监控）**

- `train_loss_x1`：原子结构去噪损失
- `train_pos_loss_x1`：坐标损失
- `train_feature_loss_x1`：原子类型损失
- `train_bond_loss_x1`：键类型损失
- `train_loss_x4`：药效团去噪损失

### **2. DPO特有指标（新增）**

| 指标 | 含义 | 期望值 | 诊断 |
|------|------|-------|------|
| `implicit_acc` | 模型是否正确偏好winner | >0.6 | <0.5说明DPO无效 |
| `model_loss_diff` | 模型在winner vs loser的损失差 | <0（负值） | 负值越大越好 |
| `ref_loss_diff` | 参考模型的损失差 | 基线对比 | 应该比model_diff大 |
| `dpo_loss` | DPO目标函数值 | 持续下降 | 上升说明不稳定 |
| `avg_score_winner` | winner平均得分 | 持续上升 | 反映生成质量 |
| `avg_score_loser` | loser平均得分 | - | 对比基线 |
| `score_diff` | 平均分差 | 稳定或增加 | 分差太小说明采样质量差 |

### **3. 采样效率指标**

- `valid_pairs_ratio`：有效偏好对比例（分差>阈值）
- `sampling_time`：采样耗时
- `pairs_per_hour`：每小时生成偏好对数

---

#### **非紧急任务 (后续优化)**

1.  **超参数精调**
    *   `beta_dpo`, 损失权重, 采样数量等。

3.  **高级采样策略**
    *   困难样本优先、采样比例调度等。初期固定一个小的采样比例（如2-5%）即可。

4.  **性能优化**
    *   偏好对数据缓存到磁盘、DDIM加速采样等。

- [ ] 如何提高训练效率？如何支持最大程度的并行

主进程（GPU）：标准训练 + DPO训练
     ↓
采样进程池（4 workers, CPU/GPU混合）：后台生成偏好对
     ↓
共享队列：异步通信，不阻塞主训练



混合精度训练（如果硬件支持）
trainer = pl.Trainer(precision="16-mixed")

梯度累积（减少通信开销）
accumulate_grad_batches = 4

避免在训练循环中做采样（移到epoch间隙）

- 偏好对存储：使用 HDF5 或 memory-mapped numpy
- 边际分布缓存：已实现（cached_marginals/）
- 评估结果缓存：避免重复计算Shepherd Score

##### Multiple Train Dataloader

##### 分层采样配置：

小分子 (10-28原子)：
├─ 采样比例：30%
├─ 每分子生成：4个候选
├─ 选择策略：均匀随机采样
└─ 原因：模型表现好，保证基础覆盖

中等分子 (29-35原子)：
├─ 采样比例：50% ⭐ 重点
├─ 每分子生成：4个候选
├─ 选择策略：基于loss的重要性采样
└─ 原因：接近OOD边界，需要重点提升

大分子 (36-70原子)：
├─ 采样比例：20%
├─ 每分子生成：6个候选 ⭐ 多生成
├─ 选择策略：只采样loss最高的Top-K
└─ 原因：OOD区域，高难度样本，多样性采样

实现关键：
1. 在每个epoch结束时，记录每个分子的训练损失
2. 根据原子数分桶：small/medium/large
3. 在每个桶内，按照不同策略采样
4. 对35+原子的分子，优先采样loss高的

---

### 三、优化后的训练流程（快速启动版）

#### 准备工作

1.  **修改LightningModule**：
2.  **配置参数**：
#### 在线采样（每个DPO Epoch开始前）

*   根据固定的采样比例（例如5%），从200万数据中随机选出一部分。
*   使用**当前`self.model`对这些数据进行采样、评分、构建偏好对，存入一个列表`self.preference_pairs`。
* 配置dpo权重超参数，初步设定40%到60%，每次epoc增加5%

#### **混合训练

*   **数据**：你需要一个能同时提供“标准数据”和“偏好对数据”的DataLoader。
*   **损失计算**：
    1.  **动态DPO权重**：`current_dpo_weight = ramp_up_function(self.current_epoch)`
    2.  **在`training_step`中**：
        *   **如果是一个标准batch**：
            `loss = standard_loss(...)`
        *   **如果是一个DPO batch**：
            `loss_std = standard_loss(on_winner_sample)`
            `loss_dpo = dpo_loss(...)`
            `loss = (1 - current_dpo_weight) * loss_std + current_dpo_weight * loss_dpo`
    3.  反向传播更新`self.model`的参数。`self.ref_model`的参数因为被冻结，不会被更新。
