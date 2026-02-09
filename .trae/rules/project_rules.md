分子生成模型训练效果评估方案

1. 分子生成任务

    - 使用训练完成的模型参数文件 last.ckpt 进行分子生成

    - 生成过程中需固定条件变量x2和x3，生成x1和x4，评估x1 (原子图模态)和x4 (药效团模态)的生成效果

2. 评估流程

    2.1 化学结构合理性评估（x1维度）

        - 使用函数`from shepherd.extract import create_rdkit_molecule`

        - 将inference输出的sample_dict输入`create_rdkit_molecule`获取mol对象

        - 判定标准：

        - 若mol为None，判定为不合理结构

        - 若结构合理，Chem.MolToSmiles(mol)应能输出有效SMILES字符串

    2.2 分子构象正确性评估（x4维度）

        - 采用shepherd score xTB方法

        - 评估步骤：

            1. 对生成的三维坐标进行xTB弛豫

            2. 计算弛豫后构象与生成pos的RMSD值

            3. RMSD阈值设定为2.0Å，低于此值视为构象合理

    2.3 条件一致性评估（P(x1,x4|x2,x3)）

        - 评估步骤：

            1. 参考 RUNME_conditional_generation_MOSESaq.ipynb 示例，对每个表面条件生成N=100个分子

            2. 使用专业工具进行质子化处理

            3. 生成表面点云和静电势

            4. 计算与条件表面的相似度指标

            5. 选取相似度最高的分子作为评估结果

训练参数文件：training/parameters/params_x1x3x4_diffusion_mosesaq_20240824.py

训练检查点文件：training/jobs/x1x3x4_diffusion_mosesaq_20240824/last.ckpt
注意：确认使用该检查点模型文件作为唯一指定的模型文件路径，禁止调用其他模型文件。

训练核心配置文件：training/new_train.py

模型配置文件：src/shepherd/model/model.py

模型训练实际过程文件：src/shepherd/lightning_module.py

评估、生成相关脚本、日志请放于文件夹：evaluation

数据文件：
1. data/conformers/np/molblock_charges_NPs.pkl
2. data/conformers/pdb/molblock_charges_pdb_lowestenergy.pkl
3. data/conformers/fragment_merging/fragment_merge_condition.pickle

代码质量注意：
1. 中间分子文件不要保存为pkl，会有依赖库，直接存含有numpy或者tensor的字典 列表，禁止使用pkl格式保存字典。

1. 代码结构优化：按照功能逻辑重新组织代码模块顺序，确保相关功能集中排列

2. 注释规范：为每个主要功能模块添加清晰的中英文注释，说明其作用和实现原理

3. 配置集中管理：将所有配置参数统一移至文件开头区域，便于查找和修改

4. 代码风格统一：遵循PEP8规范，保持一致的命名和格式风格。注释和日志统一使用中文。尽可能完整和详细

5. 功能一致性：确保优化后的代码与原代码在关键逻辑和功能表现上完全一致

6. 可读性提升：通过合理的空行、段落划分和代码分组提高整体可读性

7. 出现过长的模块，请单独分离出一个文件，文件名与模块功能相关，注释内容详细。然后在主文件中进行调用。

所有代码注释和系统日志必须统一使用中文编写，要求内容尽可能完整且详细。
注释内容应包括但不限于：功能说明、参数解释、实现逻辑、注意事项等关键信息。日志记录需包含完整的上下文信息，确保可追溯性和可读性。

严格禁止生成过程中使用“虚拟”“模拟”的方式，代替真实的代码逻辑进行填充。必须使用真实的代码逻辑。