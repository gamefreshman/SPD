### **一、 整体架构**

#### **1. 训练模式 (通过参数配置)**
*   **模式一：标准训练**：一个 epoch 中只进行标准的去噪训练。
*   **模式二：混合训练**：一个 epoch 中同时进行标准去噪训练和 DPO 偏好对训练。

#### **2. 数据加载器 (`multiple_train_dataloader`)**
*   需要一个能同时提供“标准数据”和“偏好对数据”的 DataLoader，以支持混合训练模式。

---

### **二、 核心实现步骤**

#### **1. 加载固定的参考模型 (`ref_model`)**
*   在 `LightningModule` 的 `__init__` 或 `setup` 方法中，加载预训练好的模型权重到 `self.ref_model`。
*   **立即冻结**：加载后，必须立即将其设为评估模式并冻结其参数，确保在训练中不被更新。
    ```python
    self.ref_model.eval()
    for param in self.ref_model.parameters():
        param.requires_grad = False
    ```

#### **2. 实现 DPO Score (用于评估和构建偏好对)**
*   **确定重要的参数指标**：
    *   `rmsd` (均方根偏差)
    *   `strain_energy` (应变能)
    *   `LogP` (脂水分配系数)
    *   `MQED` (药物相似性定量评估)
    *   `is valid` (分子有效性)
    *   `sims_surf_upper_bound` (表面相似性)
    *   `sims_esp_upper_bound` (静电势相似性)
*   **实现逻辑**：
    *   参考 `evaluation/main.ipynb`，了解如何调用已有的ShepherdScore评估函数。
    *   在生成的四个分子中，只保留差距分数最大的两个，形成 (winner, loser) 偏好对。
    *   如果分子间的评分差距过低，则不将其加入偏好对列表，以保证偏好信号的强度。

#### **3. 实现在线采样与偏好对构建**
*   **采样函数**：参考 `evaluation/main.ipynb` 的逻辑，编写一个函数。该函数输入一个完整的分子和当前模型 `model`，输出 4 个新生成的分子。
*   **构建偏好对列表**：调用 Shepherd Score 对生成的分子进行评估，构建 (winner, loser) 偏好对。
*   **数据利用与筛选策略**：
    *   **数据重用**：保留 50% 上一轮的偏好对，与 50% 新生成的偏好对混合，以提高数据利用率。
    *   **保留学习难度高的样本**：
        *   保留 `implicit_acc < 0.6` 的偏好对，这意味着模型还没学好这些样本。
        *   保留训练损失（loss）仍然很高的偏好对。

#### **4. 实现 DPO (Direct Preference Optimization) 损失函数**
这是核心代码，需严格按照公式实现，并同时处理连续和离散变量。
*   **关键**：在计算 winner 和 loser 的损失时，**共享噪声和时间步 `t`** 至关重要。
*   **连续变量 (坐标)**: 使用 Noise Matching (噪声匹配) 损失。
*   **离散变量 (原子/键类型)**: 使用 x0 Matching (原始数据匹配) 损失。

##### **DPO 权重 Ramp-up 策略**
在 `LightningModule` 中，根据当前 epoch 动态计算 DPO 权重，实现预热策略。
```python
def get_dpo_weight(epoch):
    if epoch < 5:
        return 0.1 * (epoch + 1)  # 线性增长
    else:
        return 0.5  # 达到稳定期
```

##### **损失计算公式**

**1. 标准去噪损失 (Standard Loss)**
*   将偏好对中的 winner 样本视作普通训练数据。
*   计算其标准的去噪损失：`loss_std = standard_loss(model, x1_w, x4_w)`。

**2. DPO 损失 (连续变量)**
```python
# 核心是比较模型预测与参考模型预测的差异
model_loss_w = MSE(noise_w_pred, noise)
model_loss_l = MSE(noise_l_pred, noise)
model_diff = model_loss_w - model_loss_l

ref_loss_w = MSE(noise_w_ref, noise)
ref_loss_l = MSE(noise_l_ref, noise)
ref_diff = ref_loss_w - ref_loss_l

inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)
loss_dpo_continuous = -torch.log(torch.sigmoid(inside_term))
```

**3. DPO 损失 (离散变量)**
```python
model_loss_w = CrossEntropy(logits_w, x_w_true)
model_loss_l = CrossEntropy(logits_l, x_l_true)
model_diff = model_loss_w - model_loss_l

ref_loss_w = CrossEntropy(logits_w_ref, x_w_true)
ref_loss_l = CrossEntropy(logits_l_ref, x_l_true)
ref_diff = ref_loss_w - ref_loss_l

inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)
loss_dpo_discrete = -torch.log(torch.sigmoid(inside_term))
```

**4. 总 DPO 损失**
`loss_dpo_total = loss_dpo_continuous + loss_dpo_discrete`

**5. 最终混合损失**
```python
# 这里的 0.7 和 0.3 可以用动态 DPO 权重代替
loss_final = (1 - dpo_weight) * loss_std + dpo_weight * loss_dpo_total
```

#### **5. 改造训练循环 (`training_step`)**
这是最复杂的修改，需要处理两种不同类型的数据。

*   **数据来源**：DataLoader 会同时提供“标准 batch”和“DPO batch”。
*   **损失计算逻辑**：
    1.  **获取动态 DPO 权重**：`current_dpo_weight = ramp_up_function(self.current_epoch)`
    2.  **在 `training_step` 中进行判断**：
        *   **如果是一个标准 batch**：
            `loss = standard_loss(...)`
        *   **如果是一个 DPO batch**：
            `loss_std = standard_loss(on_winner_sample)`
            `loss_dpo = dpo_loss(...)`
            `loss = (1 - current_dpo_weight) * loss_std + current_dpo_weight * loss_dpo`
    3.  对最终的 `loss` 进行反向传播，更新 `self.model` 的参数。`self.ref_model` 的参数因被冻结而不会更新。

---

### **三、 优化后的训练流程**

#### **准备工作**
1.  **修改 LightningModule**：完成上述所有代码实现。
2.  **配置参数**：设定 DPO 权重、采样比例、损失函数超参数 `beta_dpo` 等。初步设定 DPO 权重从 40% 到 60%，每个 epoch 增加 5%。

#### **在线采样 (每个 DPO Epoch 开始前)**
1.  根据固定的采样比例（例如 5%），从 200 万的训练数据中随机选择一部分分子作为种子。
2.  使用 **当前** `self.model` 对这些种子数据进行采样、评分、构建偏好对。
3.  将生成的偏好对存入一个列表 `self.preference_pairs`，供本 epoch 的 DPO 训练使用。

---

### **四、 监控与评估**

#### **1. 标准训练指标 (继续监控)**
*   `train_loss_x1`：原子结构去噪损失
*   `train_pos_loss_x1`：坐标损失
*   `train_feature_loss_x1`：原子类型损失
*   `train_bond_loss_x1`：键类型损失
*   `train_loss_x4`：药效团去噪损失

#### **2. DPO 特有指标 (新增)**

| 指标 | 含义 | 期望值 | 诊断分析 |
| :--- | :--- | :--- | :--- |
| `implicit_acc` | 模型是否正确偏好 winner | > 0.6 | < 0.5 说明 DPO 未生效或模型能力不足 |
| `model_loss_diff` | 模型在 (winner vs loser) 上的损失差 | < 0 (负值) | 负值越大，说明模型越偏好 winner |
| `ref_loss_diff` | 参考模型在 (winner vs loser) 上的损失差 | 基准值 | 用于与 `model_loss_diff` 对比 |
| `dpo_loss` | DPO 目标函数值 | 持续下降 | 如果上升，说明训练不稳定 |
| `avg_score_winner` | winner 样本的平均 Shepherd Score | 持续上升 | 直接反映生成分子的质量 |
| `avg_score_loser` | loser 样本的平均 Shepherd Score | - | 作为对比基线 |
| `score_diff` | winner 和 loser 的平均分差 | 稳定或增加 | 分差太小说明采样质量差或评估函数失效 |

#### **3. 采样效率指标**
*   `valid_pairs_ratio`：有效偏好对（分差 > 阈值）的比例。
*   `sampling_time`：每次在线采样过程的耗时。
*   `pairs_per_hour`：每小时能生成的有效偏好对数量。

---

### **五、 训练效率与性能优化**

#### **1. 并行化策略**
为了避免采样过程阻塞主训练循环，采用多进程架构：
*   **主进程 (GPU)**：执行标准的模型训练和 DPO 训练。
*   **采样进程池 (4 个 workers, CPU/GPU 混合)**：在后台异步地进行分子生成和评估，生成偏好对。
*   **共享队列**：主进程和采样进程通过共享队列进行异步通信，实现非阻塞的数据交换。

#### **2. 训练加速技巧**
*   **混合精度训练**：如果硬件支持，开启 FP16/BF16。`trainer = pl.Trainer(precision="16-mixed")`
*   **梯度累积**：减少 GPU 间的通信开销，变相扩大 batch size。`accumulate_grad_batches = 4`
*   **异步采样**：避免在 `training_step` 中进行采样，将其移到 epoch 之间（如 `on_epoch_start` 钩子）。

#### **3. IO 与计算优化**
*   **偏好对存储**：使用 HDF5 或 memory-mapped numpy 文件存储大量的偏好对数据，避免内存溢出。
*   **边际分布缓存**：已实现 (`cached_marginals/`)，加速计算。
*   **评估结果缓存**：缓存 Shepherd Score 的计算结果，避免对相同分子重复评估。

**实现关键**：
1.  在每个 epoch 结束时，记录每个训练样本的损失。
2.  根据原子数将样本分到 `small/medium/large` 三个桶中。
3.  在每个桶内，根据预设的策略和比例进行采样。
4.  对于原子数 > 35 的分子，优先采样损失最高的样本。

---

### **六、 非紧急任务 (后续优化)**
1.  **超参数精调**：对 `beta_dpo`、损失权重、采样数量等进行细致调整。
2.  **高级采样调度**：实现动态采样比例，例如随着训练进程调整不同分桶的采样率。
3.  **性能分析与瓶颈优化**：使用 profiler 等工具进一步分析并优化性能。

