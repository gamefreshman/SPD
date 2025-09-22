"""
分子生成模型评估管道类库

本文件包含用于评估生成分子质量和性能的完整评估管道。主要功能包括：

1. 分子构象评估 (ConfEval)：
   - 基于RDKit和xTB的分子有效性验证
   - 分子结构优化和应变能计算
   - 2D分子图性质计算（SA分数、QED、logP等）
   - RMSD对齐和一致性检查

2. 一致性评估 (ConsistencyEval)：
   - 联合生成分子特征的一致性评估
   - 3D相似性评分函数（表面、静电势、药效团）
   - 生成特征与重新生成特征的自一致性检查
   - 优化前后分子结构的一致性分析

3. 条件生成评估 (ConditionalEval)：
   - 基于参考分子的条件生成评估
   - 表面、静电势、药效团相似性计算
   - 最优对齐算法和相似性评分

4. 评估管道类：
   - UnconditionalEvalPipeline：无条件生成分子批量评估
   - ConditionalEvalPipeline：条件生成分子批量评估
   - ConsistencyEvalPipeline：一致性评估批量处理

核心算法特点：
- 支持多进程并行计算，提高评估效率
- 集成xTB量子化学计算，确保分子能量准确性
- 基于高斯重叠函数的3D相似性评分
- 完整的异常处理机制，保证评估稳定性
- 支持多种分子表示形式和特征类型

性能优化：
- 使用numpy向量化操作提高计算速度
- 临时文件管理避免内存溢出
- 批量处理减少I/O开销
- 可配置的并行度控制

作者：Shepherd Score团队
版本：1.0
更新日期：2024
"""

# ============================================================================
# 标准库导入
# ============================================================================
import sys                          # 系统相关参数和函数
import os                           # 操作系统接口
from typing import Union, List, Tuple, Optional  # 类型提示支持
from pathlib import Path            # 面向对象的文件系统路径
from tqdm import tqdm              # 进度条显示
from copy import deepcopy          # 深拷贝功能
import itertools                   # 高效循环迭代工具
from importlib.metadata import distributions  # 包元数据访问

# ============================================================================
# 科学计算库导入
# ============================================================================
import numpy as np                 # 数值计算基础库
import pandas as pd               # 数据分析和处理库
from rdkit import Chem            # RDKit化学信息学核心模块

# ============================================================================
# RDKit SA分数模块动态导入
# 根据RDKit安装方式选择不同的导入路径，确保兼容性
# ============================================================================
if any(d.metadata["Name"] == 'rdkit' for d in distributions()):
    from rdkit.Contrib.SA_Score import sascorer  # 标准安装路径
else:
    # Conda环境下的备用路径
    sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
    from SA_Score import sascorer

# ============================================================================
# RDKit分子性质和对齐模块
# ============================================================================
from rdkit.Chem import QED, Crippen, Lipinski, rdFingerprintGenerator  # 分子性质计算
from rdkit.Chem.rdMolAlign import GetBestRMS, AlignMol                 # 分子对齐算法
from rdkit.DataStructs import TanimotoSimilarity                       # Tanimoto相似性计算

# ============================================================================
# Shepherd Score内部模块导入
# ============================================================================
# 数据转换工具
from shepherd_score.evaluations.utils.convert_data import extract_mol_from_xyz_block, get_mol_from_atom_pos 

# 评分常数和缩放参数
from shepherd_score.score.constants import ALPHA, LAM_SCALING

# 构象生成和优化模块
from shepherd_score.conformer_generation import optimize_conformer_with_xtb_from_xyz_block, single_point_xtb_from_xyz

# 分子容器和分子对类
from shepherd_score.container import Molecule, MoleculePair

# 3D相似性评分函数
from shepherd_score.score.gaussian_overlap_np import get_overlap_np           # 高斯重叠评分
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np  # 静电势重叠评分
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np # 药效团重叠评分

# ============================================================================
# 全局变量和常量定义
# ============================================================================
# 随机数生成器：用于可重现的随机采样和评估
RNG = np.random.default_rng()

# Morgan指纹生成器：用于分子相似性计算
# radius=3: 指纹半径，控制原子环境的大小
# includeChirality=True: 包含手性信息，提高指纹精度
morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, includeChirality=True)

# 临时目录配置：用于xTB计算的临时文件存储
# 优先使用环境变量TMPDIR，否则使用当前目录
TMPDIR = Path('./')
if 'TMPDIR' in os.environ:
    TMPDIR = Path(os.environ['TMPDIR'])


# ============================================================================
# 分子构象评估基础类
# ============================================================================
class ConfEval:
    """
    生成分子构象评估管道基础类
    
    该类提供了对单个生成分子构象的全面评估功能，包括：
    1. 化学结构有效性验证（RDKit管道）
    2. 量子化学计算验证（xTB单点能和几何优化）
    3. 2D分子图性质计算（SA分数、QED、logP等）
    4. 3D构象一致性评估（应变能、RMSD）
    
    核心评估流程：
    - 原子坐标 → xyz格式 → RDKit分子对象
    - xTB单点能计算和几何优化
    - 优化前后分子图一致性检查
    - 分子性质和指纹计算
    """

    def __init__(self,
                 atoms: np.ndarray,
                 positions: np.ndarray,
                 solvent: Optional[str] = None,
                 num_processes: int = 1):
        """
        初始化分子构象评估器
        
        对单个生成的分子构象进行全面评估，包括RDKit有效性检查、
        xTB量子化学计算和优化、2D分子图性质计算等。
        
        自动将优化后的结构与原始结构进行RDKit RMSD对齐。
        
        参数说明
        --------
        atoms : np.ndarray, shape (N,) 或 (N,M)
            生成分子的原子序数数组或one-hot编码
            - (N,): 原子序数，如[6, 1, 1, 1]表示甲烷
            - (N,M): one-hot编码，M为元素种类数
        positions : np.ndarray, shape (N,3)
            生成分子原子的3D坐标，单位为埃(Å)
        solvent : str, optional
            xTB计算使用的溶剂类型，如'water'、'dmso'等
            None表示气相计算
        num_processes : int, default=1
            xTB优化和RDKit RMSD对齐使用的处理器数量
            
        属性说明
        --------
        优化前属性：
        - xyz_block: xyz格式的分子结构
        - mol: RDKit分子对象
        - smiles: SMILES字符串
        - molblock: MOL格式分子块
        - energy: xTB单点能(Hartree)
        - partial_charges: 原子偏电荷
        
        优化后属性：
        - xyz_block_post_opt: 优化后xyz结构
        - mol_post_opt: 优化后RDKit分子对象
        - smiles_post_opt: 优化后SMILES
        - molblock_post_opt: 优化后MOL块
        - energy_post_opt: 优化后能量
        - partial_charges_post_opt: 优化后偏电荷
        
        有效性标志：
        - is_valid: 原始结构是否有效
        - is_valid_post_opt: 优化后结构是否有效
        - is_graph_consistent: 优化前后分子图是否一致
        
        2D分子性质：
        - SA_score: 合成可达性分数(0-10，越小越易合成)
        - QED: 类药性评分(0-1，越高越好)
        - logP: 脂水分配系数
        - fsp3: sp3碳原子比例
        - morgan_fp: Morgan分子指纹
        
        3D构象性质：
        - strain_energy: 应变能(优化前后能量差)
        - rmsd: 优化前后结构RMSD(Å)
        """
        # ====================================================================
        # 初始化所有属性为None，确保属性存在性
        # ====================================================================
        # 优化前分子结构属性
        self.xyz_block = None              # xyz格式分子结构字符串
        self.mol = None                    # RDKit分子对象
        self.smiles = None                 # SMILES字符串表示
        self.molblock = None               # MOL格式分子块
        self.energy = None                 # xTB单点能(Hartree)
        self.partial_charges = None        # 原子偏电荷数组

        # 计算参数设置
        self.solvent = solvent             # 溶剂类型(用于xTB计算)
        self.charge = 0                    # 分子总电荷(默认中性)

        # 优化后分子结构属性
        self.xyz_block_post_opt = None     # 优化后xyz结构
        self.mol_post_opt = None           # 优化后RDKit分子对象
        self.smiles_post_opt = None        # 优化后SMILES
        self.molblock_post_opt = None      # 优化后MOL块
        self.energy_post_opt = None        # 优化后能量
        self.partial_charges_post_opt = None  # 优化后偏电荷

        # 有效性和一致性标志
        self.is_valid = False              # 原始结构是否化学有效
        self.is_valid_post_opt = False     # 优化后结构是否有效
        self.is_graph_consistent = False   # 优化前后分子图是否一致

        # 优化前2D分子图性质
        self.SA_score = None               # 合成可达性分数(0-10)
        self.QED = None                    # 类药性评分(0-1)
        self.logP = None                   # 脂水分配系数
        self.fsp3 = None                   # sp3碳原子比例
        self.morgan_fp = None              # Morgan分子指纹

        # 优化后2D分子图性质
        self.SA_score_post_opt = None      # 优化后SA分数
        self.QED_post_opt = None           # 优化后QED评分
        self.logP_post_opt = None          # 优化后logP
        self.fsp3_post_opt = None          # 优化后fsp3比例
        self.morgan_fp_post_opt = None     # 优化后Morgan指纹

        # 3D构象一致性评估指标
        self.strain_energy = None          # 应变能(优化前后能量差)
        self.rmsd = None                   # 优化前后结构RMSD

        # ====================================================================
        # 步骤1-2: 原子坐标转换为分子对象
        # ====================================================================
        # 将原子序数和坐标转换为xyz格式，再构建RDKit分子对象
        # 同时自动检测分子电荷状态
        self.mol, self.charge, self.xyz_block = get_mol_from_atom_pos(atoms=atoms, positions=positions)

        # ====================================================================
        # 步骤3: 初始构象的xTB单点能计算
        # ====================================================================
        # 使用xTB方法计算初始构象的单点能和原子偏电荷
        # 这是验证分子化学合理性的重要步骤
        try:
            self.energy, self.partial_charges = single_point_xtb_from_xyz(
                xyz_block=self.xyz_block,
                solvent=self.solvent,          # 溶剂环境
                charge=self.charge,            # 分子电荷
                num_cores=num_processes,       # 并行计算核数
                temp_dir=TMPDIR               # 临时文件目录
            )
            self.partial_charges = np.array(self.partial_charges)  # 转换为numpy数组
        except Exception as e:
            # xTB计算失败时保持属性为None，后续会标记为无效
            pass
        
        # 判断初始结构有效性：需要同时满足RDKit解析成功和xTB计算成功
        self.is_valid = self.mol is not None and self.partial_charges is not None
        
        if self.is_valid:
            # 生成SMILES字符串(移除氢原子以标准化)
            self.smiles = Chem.MolToSmiles(Chem.RemoveHs(self.mol))
            # 生成MOL格式分子块(包含3D坐标信息)
            self.molblock = Chem.MolToMolBlock(self.mol)

        # ====================================================================
        # 步骤4-5: xTB几何优化和优化后结构验证
        # ====================================================================
        # 使用xTB方法对分子构象进行几何优化，获得能量最小化结构
        try:
            xtb_out = optimize_conformer_with_xtb_from_xyz_block(
                self.xyz_block,
                solvent=self.solvent,
                num_cores=num_processes,
                charge=self.charge,
                temp_dir=TMPDIR
            )
            # 解包优化结果：优化后结构、能量、偏电荷
            self.xyz_block_post_opt, self.energy_post_opt, self.partial_charges_post_opt = xtb_out
            self.partial_charges_post_opt = np.array(self.partial_charges_post_opt)

            # 从优化后的xyz结构重新构建RDKit分子对象
            self.mol_post_opt = extract_mol_from_xyz_block(
                xyz_block=self.xyz_block_post_opt,
                charge=self.charge
            )
        except Exception as e:
            # 优化失败时保持相关属性为None
            pass

        # 判断优化后结构有效性
        self.is_valid_post_opt = self.mol_post_opt is not None and self.partial_charges_post_opt is not None

        # ====================================================================
        # 步骤6: 分子图一致性检查和结构对齐
        # ====================================================================
        if self.is_valid and self.is_valid_post_opt:
            # 通过比较SMILES字符串检查优化前后分子图是否一致
            # 这确保优化过程没有改变化学键连接性
            self.is_graph_consistent = Chem.MolToSmiles(self.mol) == Chem.MolToSmiles(self.mol_post_opt)
            
            # 使用RDKit进行分子结构对齐，为后续RMSD计算做准备
            mol_atom_ids = list(range(self.mol.GetNumAtoms()))
            mol_post_opt_atom_ids = list(range(self.mol_post_opt.GetNumAtoms()))
            AlignMol(prbMol=self.mol_post_opt, refMol=self.mol, 
                    atomMap=[i for i in zip(mol_post_opt_atom_ids, mol_atom_ids)])

        if self.is_valid_post_opt:
            # 生成优化后结构的SMILES和MOL块
            self.smiles_post_opt = Chem.MolToSmiles(Chem.RemoveHs(self.mol_post_opt))
            self.molblock_post_opt = Chem.MolToMolBlock(self.mol_post_opt)

        # ====================================================================
        # 步骤7: 应变能计算
        # ====================================================================
        # 应变能 = 初始能量 - 优化后能量，反映初始构象的不稳定程度
        if self.energy is not None and self.energy_post_opt is not None:
            self.strain_energy = self.energy - self.energy_post_opt

        # ====================================================================
        # 步骤8: RMSD计算
        # ====================================================================
        # 计算优化前后结构的均方根偏差，评估构象变化程度
        if self.is_graph_consistent:
            # 使用深拷贝避免修改原始分子对象
            mol_copy = deepcopy(Chem.RemoveHs(self.mol))
            mol_post_opt_copy = deepcopy(Chem.RemoveHs(self.mol_post_opt))
            self.rmsd = GetBestRMS(prbMol=mol_copy, refMol=mol_post_opt_copy, 
                                 numThreads=num_processes)

        # ====================================================================
        # 步骤9: 优化前2D分子性质计算
        # ====================================================================
        if self.is_valid:
            # SA分数：合成可达性评分，范围0-10，越小表示越容易合成
            self.SA_score = sascorer.calculateScore(Chem.RemoveHs(self.mol))
            # QED：定量药物相似性评分，范围0-1，越高表示越像药物
            self.QED = QED.qed(self.mol)
            # logP：脂水分配系数，影响药物的ADMET性质
            self.logP = Crippen.MolLogP(self.mol)
            # fsp3：sp3碳原子比例，影响分子的3D性质和药物相似性
            self.fsp3 = Lipinski.FractionCSP3(self.mol)
            # Morgan指纹：用于分子相似性比较的环形指纹
            self.morgan_fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(self.mol))
        
        # ====================================================================
        # 步骤10: 优化后2D分子性质计算
        # ====================================================================
        if self.is_valid_post_opt:
            # 计算优化后结构的相同2D性质，用于比较优化前后的变化
            self.SA_score_post_opt = sascorer.calculateScore(Chem.RemoveHs(self.mol_post_opt))
            self.QED_post_opt = QED.qed(self.mol_post_opt)
            self.logP_post_opt = Crippen.MolLogP(self.mol_post_opt)
            self.fsp3_post_opt = Lipinski.FractionCSP3(self.mol_post_opt)
            self.morgan_fp_post_opt = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(self.mol_post_opt))


    def to_pandas(self):
        """
        将存储的所有属性转换为pandas Series格式
        
        该方法将ConfEval对象的所有属性（包括分子结构、能量、性质等）
        转换为pandas Series，便于数据分析、可视化和导出。
        
        转换的属性包括：
        - 分子结构信息：xyz_block, mol, smiles, molblock等
        - 能量信息：energy, energy_post_opt, strain_energy等  
        - 有效性标志：is_valid, is_valid_post_opt, is_graph_consistent
        - 2D分子性质：SA_score, QED, logP, fsp3, morgan_fp等
        - 3D构象信息：rmsd, partial_charges等
        
        参数说明
        --------
        self : ConfEval对象
            包含所有评估结果的ConfEval实例
            
        返回值
        -------
        pd.Series
            包含所有属性的pandas Series对象，索引为属性名，值为属性值
            便于后续的数据分析、统计和可视化操作
            
        使用示例
        --------
        >>> conf_eval = ConfEval(atoms, positions)
        >>> series = conf_eval.to_pandas()
        >>> print(series['is_valid'])  # 查看结构有效性
        >>> print(series['SA_score'])  # 查看合成可达性分数
        """
        # 初始化全局属性字典，用于存储所有对象属性
        global_attrs = {}
        
        # 遍历对象的所有属性，将其添加到字典中
        # __dict__包含对象的所有实例属性
        for key, value in self.__dict__.items():
            global_attrs[key] = value
        
        # 将属性字典转换为pandas Series
        # Series提供了便于数据分析的接口和方法
        series_global = pd.Series(global_attrs)
        
        return series_global


class ConsistencyEval(ConfEval):
    """
    分子特征一致性评估类
    
    该类用于评估联合生成的分子及其特征之间的一致性，通过3D相似性评分函数
    来量化生成的分子结构与其对应特征（表面、静电势、药效团）的匹配程度。
    
    继承自ConfEval类，首先对生成的分子进行构象评估，然后进行特征一致性分析。
    
    核心功能：
    - 分子表面一致性评估：比较生成的表面点云与重新计算的表面
    - 静电势一致性评估：比较生成的静电势与重新计算的静电势
    - 药效团一致性评估：比较生成的药效团特征与重新计算的药效团
    - 结构优化后一致性：评估xTB优化后结构与原始特征的一致性
    
    重要假设：
    1. 表面相似性的高斯宽度参数(alpha)针对探针半径1.2Å进行了拟合
    2. 静电势相似性的权重参数(lam)设置为0.3，已针对假设1进行测试
    """
    def __init__(self,
                 atoms: np.ndarray,
                 positions: np.ndarray,
                 surf_points: Optional[np.ndarray] = None,
                 surf_esp: Optional[np.ndarray] = None,
                 pharm_feats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 probe_radius: float = 1.2,
                 num_processes: int = 1):
        """
        初始化分子特征一致性评估对象
        
        该方法首先调用父类ConfEval进行基础的分子构象评估，然后进行特征一致性分析。
        必须提供atoms和positions，以及至少一种用于相似性评分的特征。
        
        评估流程：
        1. 基础构象评估（继承自ConfEval）
        2. 特征验证和参数设置
        3. 生成分子对象（原始特征 vs 重新计算特征）
        4. 计算各种一致性评分
        5. 结构优化后的一致性评估
        
        参数说明
        --------
        atoms : np.ndarray, shape (N,) 或 (N,M)
            生成分子的原子序数数组或one-hot编码
            N为原子数，M为编码维度
            
        positions : np.ndarray, shape (N,3)
            生成分子的原子坐标数组
            每行代表一个原子的x,y,z坐标
            
        surf_points : Optional[np.ndarray], shape (M,3)
            生成的分子表面点云坐标
            M为表面点数量，用于表面相似性评估
            
        surf_esp : Optional[np.ndarray], shape (M,)
            生成的分子表面静电势值
            与surf_points对应，用于静电势相似性评估
            
        pharm_feats : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            生成的药效团特征，包含三个数组：
            - pharm_types: shape (P,) 药效团类型，定义见P_TYPES常量
            - pharm_ancs: shape (P,3) 药效团锚点位置坐标
            - pharm_vecs: shape (P,3) 药效团相对锚点的单位向量
            P为药效团数量
            
        pharm_multi_vector : Optional[bool]
            是否使用多向量表示芳香环/氢键受体/氢键供体药效团
            True: 使用多向量表示，提高精度
            False: 使用单向量表示，计算更快
            None: 不进行药效团评估
            
        solvent : Optional[str]
            xTB结构优化使用的溶剂类型
            支持的溶剂类型见xTB文档，如'water', 'dmso'等
            
        probe_radius : float, default=1.2
            生成溶剂可及表面时使用的探针原子半径（单位：埃）
            默认1.2埃为氢原子的范德华半径
            
        num_processes : int, default=1
            xTB结构优化使用的处理器数量
            多进程可加速计算但需要更多内存
            
        属性说明
        --------
        初始化后会设置以下一致性评分属性：
        
        基础一致性评分（生成特征 vs 重新计算特征）：
        - sim_surf_consistent : 表面相似性评分
        - sim_esp_consistent : 静电势相似性评分  
        - sim_pharm_consistent : 药效团相似性评分
        
        优化后一致性评分（生成特征 vs 优化后重新计算特征）：
        - sim_surf_consistent_relax : 表面相似性评分
        - sim_esp_consistent_relax : 静电势相似性评分
        - sim_pharm_consistent_relax : 药效团相似性评分
        
        最优对齐一致性评分（通过最优对齐算法）：
        - sim_surf_consistent_relax_optimal : 最优表面相似性评分
        - sim_esp_consistent_relax_optimal : 最优静电势相似性评分
        - sim_pharm_consistent_relax_optimal : 最优药效团相似性评分
        
        分子对象：
        - molec : 包含生成特征的分子对象
        - molec_regen : 包含重新计算特征的分子对象
        - molec_post_opt : 包含优化后特征的分子对象
        
        评分参数：
        - alpha : 高斯重叠函数的宽度参数
        - lam : 静电势权重参数（用于对齐）
        - lam_scaled : 缩放后的静电势权重参数（用于评分）
        
        异常处理
        --------
        ValueError
            当atoms或positions不是np.ndarray类型时抛出
            当未提供任何特征（表面、静电势、药效团）时抛出
            当药效团特征维度不匹配时抛出
            
        使用示例
        --------
        >>> # 基本使用
        >>> atoms = np.array([6, 1, 1, 1, 1])  # 甲烷
        >>> positions = np.random.rand(5, 3)
        >>> surf_points = np.random.rand(100, 3)
        >>> 
        >>> eval_obj = ConsistencyEval(atoms, positions, surf_points=surf_points)
        >>> print(f"表面一致性评分: {eval_obj.sim_surf_consistent}")
        
        >>> # 包含所有特征的评估
        >>> surf_esp = np.random.rand(100)
        >>> pharm_types = np.array([0, 1])
        >>> pharm_ancs = np.random.rand(2, 3)
        >>> pharm_vecs = np.random.rand(2, 3)
        >>> pharm_feats = (pharm_types, pharm_ancs, pharm_vecs)
        >>> 
        >>> eval_obj = ConsistencyEval(
        ...     atoms, positions,
        ...     surf_points=surf_points,
        ...     surf_esp=surf_esp,
        ...     pharm_feats=pharm_feats,
        ...     pharm_multi_vector=True
        ... )
        """
        # 验证输入参数类型
        if not (isinstance(atoms, np.ndarray) or isinstance(positions, np.ndarray)):
            raise ValueError(f"Must provide `atoms` and `positions` as np.ndarrays. Instead {type(atoms)} and {type(positions)} were given.")

        # 调用父类ConfEval的初始化方法，进行基础的分子构象评估
        # 包括分子有效性检查、xTB能量计算、结构优化等
        super().__init__(atoms=atoms, positions=positions, solvent=solvent, num_processes=num_processes)

        # 初始化分子对象属性
        self.molec = None              # 包含生成特征的分子对象
        self.probe_radius = probe_radius  # 探针半径，用于表面生成
        self.molec_regen = None        # 包含重新计算特征的分子对象
        self.molec_post_opt = None     # 包含优化后特征的分子对象

        # 初始化基础一致性评分属性（生成特征 vs 重新计算特征）
        self.sim_surf_consistent = None   # 表面相似性评分
        self.sim_esp_consistent = None    # 静电势相似性评分
        self.sim_pharm_consistent = None  # 药效团相似性评分

        # 初始化优化后一致性评分属性（生成特征 vs 优化后重新计算特征）
        self.sim_surf_consistent_relax = None   # 表面相似性评分（优化后）
        self.sim_esp_consistent_relax = None    # 静电势相似性评分（优化后）
        self.sim_pharm_consistent_relax = None  # 药效团相似性评分（优化后）

        # 初始化最优对齐一致性评分属性（通过最优对齐算法）
        self.sim_surf_consistent_relax_optimal = None   # 最优表面相似性评分
        self.sim_esp_consistent_relax_optimal = None    # 最优静电势相似性评分
        self.sim_pharm_consistent_relax_optimal = None  # 最优药效团相似性评分
        
        # 处理药效团特征参数
        if pharm_feats is not None:
            # 解包药效团特征元组
            pharm_types, pharm_ancs, pharm_vecs = pharm_feats
            num_pharms = len(pharm_types)  # 药效团数量

            # 验证药效团特征维度的一致性
            if pharm_ancs.shape != (num_pharms, 3) or pharm_vecs.shape != (num_pharms, 3):
                raise ValueError(
                    f'Provided pharmacophore features do not match dimensions: pharm_types {pharm_types.shape}, pharm_ancs {pharm_ancs.shape}, pharm_vecs {pharm_vecs.shape}'
                )
        else:
            # 如果未提供药效团特征，设置为None
            pharm_types, pharm_ancs, pharm_vecs = None, None, None

        # 检查是否提供了完整的药效团特征
        has_pharm_features = (isinstance(pharm_types, np.ndarray)
                              and isinstance(pharm_ancs, np.ndarray)
                              and isinstance(pharm_vecs, np.ndarray))
        
        # 药效团特征警告检查
        if has_pharm_features and pharm_multi_vector is None:
            print('WARNING: Generated pharmacophore features provided, but `pharm_multi_vector` is None.')
            print('         Pharmacophore similarity not computed.')
        
        # 验证至少提供了一种特征用于相似性评分
        if not isinstance(surf_points, np.ndarray) and not isinstance(surf_esp, np.ndarray) and not has_pharm_features:
            raise ValueError(f'Must provide at least one of the generated representations: surface, electrostatics, or pharmacophores.')
        
        # 设置评分参数
        self.num_surf_points = len(surf_points) if surf_points is not None else None  # 表面点数量
        # 高斯宽度参数，假设探针半径为1.2Å时无半径缩放
        self.alpha = ALPHA(self.num_surf_points) if self.num_surf_points is not None else None
        self.lam = 0.3  # 探针半径1.2Å的最优lambda参数 -> 仅用于ESP对齐
        self.lam_scaled = self.lam * LAM_SCALING  # 缩放后的lambda -> 仅用于get_overlap_esp*函数

        # 生成分子的特征自一致性评估
        if self.is_valid:
            # 创建包含生成特征的分子对象
            self.molec = Molecule(
                self.mol,                                    # RDKit分子对象
                partial_charges=np.array(self.partial_charges),  # 原子部分电荷
                surface_points=surf_points,                  # 生成的表面点云
                electrostatics=surf_esp,                     # 生成的表面静电势
                pharm_types=pharm_types,                     # 生成的药效团类型
                pharm_ancs=pharm_ancs,                       # 生成的药效团锚点
                pharm_vecs=pharm_vecs,                       # 生成的药效团向量
                probe_radius=self.probe_radius               # 探针半径
            )
            
            # 创建包含重新计算特征的分子对象（用于对比）
            self.molec_regen = Molecule(
                self.mol,                                    # 同一个RDKit分子对象
                num_surf_points=self.num_surf_points,        # 表面点数量（重新计算）
                probe_radius=self.probe_radius,              # 探针半径
                partial_charges=np.array(self.partial_charges),  # 原子部分电荷
                pharm_multi_vector=pharm_multi_vector if has_pharm_features else None  # 药效团多向量设置
            )

            # 计算表面相似性评分（如果表面点存在）
            if self.molec.surf_pos is not None:
                self.sim_surf_consistent = get_overlap_np(
                    self.molec.surf_pos,        # 生成的表面点
                    self.molec_regen.surf_pos,  # 重新计算的表面点
                    alpha=self.alpha            # 高斯宽度参数
                )

            # 计算静电势相似性评分（如果静电势和表面点都存在）
            if self.molec.surf_esp is not None and self.molec.surf_pos is not None:
                self.sim_esp_consistent = get_overlap_esp_np(
                    self.molec.surf_pos, self.molec_regen.surf_pos,      # 表面点坐标
                    self.molec.surf_esp, self.molec_regen.surf_esp,      # 表面静电势值
                    alpha=self.alpha,                                    # 高斯宽度参数
                    lam=self.lam_scaled                                  # 缩放后的权重参数
                )

            # 计算药效团相似性评分（如果药效团特征完整且启用多向量）
            if has_pharm_features and pharm_multi_vector is not None:
                self.sim_pharm_consistent = get_overlap_pharm_np(
                    self.molec.pharm_types,      # 生成的药效团类型
                    self.molec_regen.pharm_types,  # 重新计算的药效团类型
                    self.molec.pharm_ancs,       # 生成的药效团锚点
                    self.molec_regen.pharm_ancs,   # 重新计算的药效团锚点
                    self.molec.pharm_vecs,       # 生成的药效团向量
                    self.molec_regen.pharm_vecs,   # 重新计算的药效团向量
                    similarity='tanimoto',       # 使用Tanimoto相似性
                    extended_points=False,       # 不使用扩展点
                    only_extended=False          # 不仅使用扩展点
                )

        # Consistency between generated molecule and relaxed structure and features
        if self.is_valid and self.is_valid_post_opt: 
            # Generate a Molecule object of relaxed structure
            self.molec_post_opt = Molecule(
                self.mol_post_opt,
                num_surf_points = self.num_surf_points,
                probe_radius=self.probe_radius,
                partial_charges = np.array(self.partial_charges_post_opt),
                pharm_multi_vector = pharm_multi_vector if has_pharm_features else None
            )

            # Score only since we already align w.r.t. RMS of the generated atomic point cloud
            if self.molec_post_opt.surf_pos is not None:
                self.sim_surf_consistent_relax = get_overlap_np(
                    self.molec.surf_pos,
                    self.molec_post_opt.surf_pos,
                    alpha=self.alpha
                )
            if self.molec_post_opt.surf_pos is not None and self.molec_post_opt.surf_esp is not None:            
                self.sim_esp_consistent_relax = get_overlap_esp_np(
                    self.molec.surf_pos, self.molec_post_opt.surf_pos,
                    self.molec.surf_esp, self.molec_post_opt.surf_esp,
                    alpha=self.alpha,
                    lam=self.lam_scaled
                )
            if isinstance(pharm_multi_vector, bool) and self.molec_post_opt.pharm_ancs is not None and self.molec.pharm_ancs is not None:
                self.sim_pharm_consistent_relax = get_overlap_pharm_np(
                    self.molec.pharm_types,
                    self.molec_post_opt.pharm_types,
                    self.molec.pharm_ancs,
                    self.molec_post_opt.pharm_ancs,
                    self.molec.pharm_vecs,
                    self.molec_post_opt.pharm_vecs,
                    similarity='tanimoto',
                    extended_points=False,
                    only_extended=False
                )

            # Alignment with scoring functions
            mp_ref_and_relaxed = MoleculePair(self.molec,
                                              self.molec_post_opt,
                                              num_surf_points=self.num_surf_points,
                                              do_center=False)
            if self.molec_post_opt.surf_pos is not None:
                self.sim_surf_consistent_relax_optimal = self._align_with_surface(mp_ref_and_relaxed=mp_ref_and_relaxed)
            if self.molec_post_opt.surf_pos is not None and self.molec_post_opt.surf_esp is not None:            
                self.sim_esp_consistent_relax_optimal = self._align_with_esp(mp_ref_and_relaxed=mp_ref_and_relaxed)
            if isinstance(pharm_multi_vector, bool) and self.molec_post_opt.pharm_ancs is not None and self.molec.pharm_ancs is not None:
                self.sim_pharm_consistent_relax_optimal = self._align_with_pharm(mp_ref_and_relaxed=mp_ref_and_relaxed)


    def _align_with_surface(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with surface.

        Returns
        -------
        float : Surface similarity score of optimally aligned molecule.
        """
        aligned_surf_points = mp_ref_and_relaxed.align_with_surf(
            self.alpha,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        )
        surf_similarity = mp_ref_and_relaxed.sim_aligned_surf
        return float(surf_similarity)
    

    def _align_with_esp(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with ESP

        Returns
        -------
        float : ESP similarity score of optimally aligned molecule.
        """
        aligned_surf_points = mp_ref_and_relaxed.align_with_esp(
            self.alpha,
            lam=self.lam,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        ) 
        esp_similarity = mp_ref_and_relaxed.sim_aligned_esp
        return float(esp_similarity)


    def _align_with_pharm(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        Align relaxed molecule to reference/target molecule with pharmacophores

        Returns
        -------
        float : Pharmacophore similarity score of optimally aligned molecule.
        """
        aligned_fit_anchors, aligned_vectors = mp_ref_and_relaxed.align_with_pharm(
            similarity='tanimoto',
            extended_points=False,
            only_extended=False,
            num_repeats=1,
            trans_init=False,
            use_jax=False
        )
        pharm_similarity = mp_ref_and_relaxed.sim_aligned_pharm
        return float(pharm_similarity)


class ConditionalEval(ConfEval):
    """
    条件生成分子的质量和相似性评估类
    
    该类继承自ConfEval，专门用于评估条件生成分子与参考分子的相似性。
    主要功能包括：
    1. 基于不同条件（表面、静电势、药效团）的分子对齐和相似性评分
    2. 多种相似性指标的计算（表面重叠、静电势匹配、药效团相似性）
    3. 优化前后分子结构的对比评估
    4. 基于特定条件的最优对齐算法
    
    核心算法特点：
    - 使用高斯重叠函数计算表面相似性
    - 基于静电势分布的分子对齐和评分
    - 药效团特征的Tanimoto相似性计算
    - 支持多种对齐策略和评分模式
    """

    def __init__(self,
                 ref_molec: Molecule,
                 atoms: np.ndarray,
                 positions: np.ndarray,
                 condition: str,
                 num_surf_points: int = 400,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 num_processes: int = 1):
        """
        初始化条件生成分子评估管道
        
        该方法继承自ConfEval，首先对生成的分子进行构象评估，然后进行条件相似性评估。
        
        重要假设：
        1. 表面相似性的高斯宽度参数(alpha)假设探针半径为1.2Å
        2. 静电势相似性的权重参数(lam)设置为0.3，该值针对假设1进行了优化
        
        参数说明：
        ----------
        ref_molec : Molecule
            参考/目标分子对象，必须包含用于条件生成的表示（表面、静电势或药效团）
        atoms : np.ndarray, shape (N,) 或 (N,M)
            生成分子的原子序数数组或独热编码
        positions : np.ndarray, shape (N,3)
            生成分子原子的三维坐标
        condition : str
            分子条件生成的类型，可选值：'surface'/'surf'、'esp'/'electrostatic'、'pharm'、'all'
            用于确定对齐策略。选择'esp'或'all'可计算ESP对齐的其他特征评分
        num_surf_points : int, 默认400
            用于相似性评分的表面点采样数量
        pharm_multi_vector : Optional[bool]
            是否使用多向量表示芳香性/氢键受体/氢键供体特征，或使用单向量
        solvent : Optional[str]
            xTB弛豫计算的溶剂类型
        num_processes : int, 默认1
            xTB弛豫计算使用的处理器数量
            
        属性说明：
        ----------
        condition : str
            标准化后的条件类型（'surface', 'esp', 'pharm', 'all'）
        sim_surf_target : float
            生成分子与目标分子的表面相似性评分
        sim_esp_target : float
            生成分子与目标分子的静电势相似性评分
        sim_pharm_target : float
            生成分子与目标分子的药效团相似性评分
        sim_*_target_relax : float
            弛豫后分子与目标分子的相似性评分（各种类型）
        sim_*_target_relax_optimal : float
            最优对齐后分子与目标分子的相似性评分
        sim_*_target_relax_esp_aligned : float
            ESP对齐后的其他特征相似性评分
        ref_molec : Molecule
            参考分子对象
        molec_regen : Molecule
            重新生成特征的分子对象
        molec_post_opt : Molecule
            优化后的分子对象
            
        异常处理：
        ----------
        ValueError : 当condition参数不在允许的值范围内时抛出
        
        使用示例：
        ----------
        >>> ref_mol = Molecule(...)  # 参考分子
        >>> atoms = np.array([6, 6, 8, 1, 1, 1])  # 原子序数
        >>> positions = np.random.rand(6, 3)  # 原子坐标
        >>> cond_eval = ConditionalEval(
        ...     ref_molec=ref_mol,
        ...     atoms=atoms,
        ...     positions=positions,
        ...     condition='esp',
        ...     num_surf_points=400
        ... )
        >>> print(f"ESP相似性: {cond_eval.sim_esp_target_relax_optimal}")
        """
        # 条件类型标准化处理
        condition = condition.lower()
        self.condition = None
        if 'surf' in condition or 'shape' in condition:
            self.condition = 'surface'  # 表面/形状条件
        elif 'esp' in condition or 'electrostatic' in condition:
            self.condition = 'esp'      # 静电势条件
        elif 'pharm' in condition:
            self.condition = 'pharm'    # 药效团条件
        elif condition == 'all':
            self.condition = 'all'      # 全部条件
        else:
            raise ValueError(f'`condition` must contain one of the following: "surf", "esp", "pharm", or "all". Instead, {condition} was given.')

        # 调用父类ConfEval的初始化方法，进行基础构象评估
        super().__init__(atoms=atoms,
                         positions=positions,
                         solvent=solvent,
                         num_processes=num_processes)

        # 初始化相似性评分属性 - 生成分子与目标分子的直接比较
        self.sim_surf_target = None   # 表面相似性评分
        self.sim_esp_target = None    # 静电势相似性评分
        self.sim_pharm_target = None  # 药效团相似性评分

        # 弛豫后分子与目标分子的相似性评分（基于RMS对齐）
        self.sim_surf_target_relax = None   # 弛豫后表面相似性
        self.sim_esp_target_relax = None    # 弛豫后静电势相似性
        self.sim_pharm_target_relax = None  # 弛豫后药效团相似性

        # 最优对齐后的相似性评分（基于指定条件的最佳对齐）
        self.sim_surf_target_relax_optimal = None   # 最优对齐表面相似性
        self.sim_esp_target_relax_optimal = None    # 最优对齐静电势相似性
        self.sim_pharm_target_relax_optimal = None  # 最优对齐药效团相似性

        # ESP对齐后的其他特征相似性评分
        self.sim_surf_target_relax_esp_aligned = None   # ESP对齐后的表面相似性
        self.sim_pharm_target_relax_esp_aligned = None  # ESP对齐后的药效团相似性
        
        # 评分参数设置
        self.num_surf_points = num_surf_points                    # 表面点数量
        self.alpha = ALPHA(self.num_surf_points)                  # 高斯宽度参数，针对探针半径1.2Å优化
        self.lam = 0.3                                           # 探针半径1.2Å的最优lambda参数 -> 仅用于ESP对齐
        self.lam_scaled = self.lam * LAM_SCALING                 # 缩放后的lambda -> 仅用于get_overlap_esp*函数

        # 分子对象存储
        self.ref_molec = ref_molec      # 参考分子对象
        self.molec_regen = None         # 重新生成特征的分子对象
        self.molec_post_opt = None      # 优化后的分子对象

        # 如果分子结构有效，进行特征重新生成和相似性计算
        if self.is_valid:
            # 使用重新生成的特征创建分子对象
            self.molec_regen = Molecule(
                self.mol,                                           # RDKit分子对象
                num_surf_points = self.num_surf_points,             # 表面点数量
                partial_charges = np.array(self.partial_charges),   # 原子部分电荷
                pharm_multi_vector = pharm_multi_vector,            # 药效团多向量标志
                probe_radius=self.ref_molec.probe_radius            # 探针半径（与参考分子保持一致）
            )

            # 计算表面相似性评分（如果表面点存在）
            if self.molec_regen.surf_pos is not None:
                self.sim_surf_target = get_overlap_np(
                    self.molec_regen.surf_pos,    # 生成分子的表面点
                    self.ref_molec.surf_pos,      # 参考分子的表面点
                    alpha=self.alpha              # 高斯宽度参数
                )

            # 计算静电势相似性评分（如果静电势和表面点都存在）
            if self.molec_regen.surf_esp is not None and self.molec_regen.surf_pos is not None:
                self.sim_esp_target = get_overlap_esp_np(
                    self.molec_regen.surf_pos, self.ref_molec.surf_pos,  # 表面点坐标
                    self.molec_regen.surf_esp, self.ref_molec.surf_esp,  # 表面静电势值
                    alpha=self.alpha,                                    # 高斯宽度参数
                    lam=self.lam_scaled                                  # 缩放后的lambda参数
                )

            # 计算药效团相似性评分（如果药效团特征存在）
            if pharm_multi_vector is not None and self.ref_molec.pharm_ancs is not None:
                self.sim_pharm_target = get_overlap_pharm_np(
                    self.molec_regen.pharm_types,   # 生成分子的药效团类型
                    self.ref_molec.pharm_types,     # 参考分子的药效团类型
                    self.molec_regen.pharm_ancs,    # 生成分子的药效团锚点
                    self.ref_molec.pharm_ancs,      # 参考分子的药效团锚点
                    self.molec_regen.pharm_vecs,    # 生成分子的药效团向量
                    self.ref_molec.pharm_vecs,      # 参考分子的药效团向量
                    similarity='tanimoto',          # 使用Tanimoto相似性
                    extended_points=False,          # 不使用扩展点
                    only_extended=False             # 不仅使用扩展点
                )

        # 弛豫结构与目标分子的相似性计算 -> 首先进行对齐
        if self.is_valid_post_opt:
            # 为弛豫后的结构生成分子对象
            self.molec_post_opt = Molecule(
                self.mol_post_opt,                                      # 弛豫后的RDKit分子对象
                num_surf_points = self.num_surf_points,                 # 表面点数量
                partial_charges = np.array(self.partial_charges_post_opt),  # 弛豫后的原子部分电荷
                pharm_multi_vector = pharm_multi_vector,                # 药效团多向量标志
                probe_radius=self.ref_molec.probe_radius                # 探针半径（与参考分子保持一致）
            )

            # 基于RMS对齐的评分计算
            # 计算弛豫后分子的表面相似性评分
            if self.molec_post_opt.surf_pos is not None:
                self.sim_surf_target_relax = get_overlap_np(
                    self.molec_post_opt.surf_pos,   # 弛豫后分子的表面点
                    self.ref_molec.surf_pos,        # 参考分子的表面点
                    alpha=self.alpha                # 高斯宽度参数
                )

            # 计算弛豫后分子的静电势相似性评分
            if self.molec_post_opt.surf_esp is not None and self.molec_post_opt.surf_pos is not None:
                self.sim_esp_target_relax = get_overlap_esp_np(
                    self.molec_post_opt.surf_pos, self.ref_molec.surf_pos,  # 表面点坐标
                    self.molec_post_opt.surf_esp, self.ref_molec.surf_esp,  # 表面静电势值
                    alpha=self.alpha,                                       # 高斯宽度参数
                    lam=self.lam_scaled                                     # 缩放后的lambda参数
                )

            # 计算弛豫后分子的药效团相似性评分
            if pharm_multi_vector is not None and self.ref_molec.pharm_ancs is not None:
                self.sim_pharm_target_relax = get_overlap_pharm_np(
                    self.molec_post_opt.pharm_types,    # 弛豫后分子的药效团类型
                    self.ref_molec.pharm_types,         # 参考分子的药效团类型
                    self.molec_post_opt.pharm_ancs,     # 弛豫后分子的药效团锚点
                    self.ref_molec.pharm_ancs,          # 参考分子的药效团锚点
                    self.molec_post_opt.pharm_vecs,     # 弛豫后分子的药效团向量
                    self.ref_molec.pharm_vecs,          # 参考分子的药效团向量
                    similarity='tanimoto',              # 使用Tanimoto相似性
                    extended_points=False,              # 不使用扩展点
                    only_extended=False                 # 不仅使用扩展点
                )

            # 根据指定条件进行对齐和评分
            mp_ref_and_relaxed = MoleculePair(self.ref_molec,          # 参考分子
                                              self.molec_post_opt,      # 弛豫后分子
                                              num_surf_points=self.num_surf_points,  # 表面点数量
                                              do_center=False)          # 不进行中心化处理
            
            # 根据条件类型进行最优对齐评分
            # 表面条件：基于表面特征进行最优对齐
            if (self.condition == 'surface' or self.condition == 'all') and self.molec_post_opt.surf_pos is not None:
                self.sim_surf_target_relax_optimal = self._align_with_surface(mp_ref_and_relaxed=mp_ref_and_relaxed)
            
            # 静电势条件：基于静电势特征进行最优对齐
            if (self.condition == 'esp' or self.condition == 'all') and self.molec_post_opt.surf_pos is not None and self.molec_post_opt.surf_esp is not None:
                self.sim_esp_target_relax_optimal = self._align_with_esp(mp_ref_and_relaxed=mp_ref_and_relaxed)
            
            # 药效团条件：基于药效团特征进行最优对齐
            if (self.condition == 'pharm' or self.condition == 'all') and isinstance(pharm_multi_vector, bool):
                self.sim_pharm_target_relax_optimal = self._align_with_pharm(mp_ref_and_relaxed=mp_ref_and_relaxed)

            # 计算ESP对齐后的表面和药效团相似性评分
            if mp_ref_and_relaxed.transform_esp is not None and self.condition in ('esp', 'all'):
                # 获取ESP对齐后的分子
                molec_post_opt_esp_aligned = mp_ref_and_relaxed.get_transformed_molecule(mp_ref_and_relaxed.transform_esp)
                
                # 创建ESP对齐后的分子对
                esp_aligned_molec_pair = MoleculePair(ref_mol=self.ref_molec,                    # 参考分子
                                                      fit_mol=molec_post_opt_esp_aligned,        # ESP对齐后的分子
                                                      num_surf_points=self.ref_molec.num_surf_points,  # 表面点数量
                                                      do_center=False)                           # 不进行中心化处理
                
                # 计算ESP对齐后的表面相似性
                self.sim_surf_target_relax_esp_aligned = esp_aligned_molec_pair.score_with_surf(
                    alpha=self.alpha,   # 高斯宽度参数
                    use='np'           # 使用NumPy实现
                )
                
                # 计算ESP对齐后的药效团相似性（如果药效团特征存在）
                if isinstance(pharm_multi_vector, bool):
                    self.sim_pharm_target_relax_esp_aligned = esp_aligned_molec_pair.score_with_pharm(
                        similarity='tanimoto',      # 使用Tanimoto相似性
                        extended_points=False,      # 不使用扩展点
                        only_extended=False,        # 不仅使用扩展点
                        use='np'                   # 使用NumPy实现
                    )


    def _align_with_surface(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        基于表面特征将弛豫后分子与参考/目标分子进行对齐。
        
        该方法使用表面点云进行分子对齐，通过优化旋转和平移变换
        来最大化两个分子表面的重叠度。
        
        参数
        ----
        mp_ref_and_relaxed : MoleculePair
            包含参考分子和弛豫后分子的分子对对象
            
        返回
        ----
        float
            最优对齐后的表面相似性评分 (0-1范围)
            
        注意
        ----
        - 使用高斯重叠函数计算表面相似性
        - 对齐过程不使用JAX加速，采用传统优化方法
        - 仅进行一次对齐尝试以提高计算效率
        """
        # 执行基于表面的分子对齐
        aligned_surf_points = mp_ref_and_relaxed.align_with_surf(
            self.alpha,         # 高斯宽度参数
            num_repeats=1,      # 对齐重复次数
            trans_init=False,   # 不使用平移初始化
            use_jax=False       # 不使用JAX加速
        )

        # 获取对齐后的表面相似性评分
        surf_similarity = mp_ref_and_relaxed.sim_aligned_surf
        return float(surf_similarity)
    

    def _align_with_esp(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        基于静电势特征将弛豫后分子与参考/目标分子进行对齐。
        
        该方法同时考虑表面形状和静电势分布进行分子对齐，
        通过优化旋转和平移变换来最大化静电势相似性。
        
        参数
        ----
        mp_ref_and_relaxed : MoleculePair
            包含参考分子和弛豫后分子的分子对对象
            
        返回
        ----
        float
            最优对齐后的静电势相似性评分 (0-1范围)
            
        注意
        ----
        - 同时考虑几何形状和静电势分布
        - lambda参数控制静电势权重
        - 对齐结果可用于后续的表面和药效团评分
        """
        # 执行基于静电势的分子对齐
        aligned_surf_points = mp_ref_and_relaxed.align_with_esp(
            self.alpha,         # 高斯宽度参数
            lam=self.lam,       # 静电势权重参数
            num_repeats=1,      # 对齐重复次数
            trans_init=False,   # 不使用平移初始化
            use_jax=False       # 不使用JAX加速
        ) 
        # 获取对齐后的静电势相似性评分
        esp_similarity = mp_ref_and_relaxed.sim_aligned_esp
        return float(esp_similarity)


    def _align_with_pharm(self, mp_ref_and_relaxed: MoleculePair) -> float:
        """
        基于药效团特征将弛豫后分子与参考/目标分子进行对齐。
        
        该方法使用药效团锚点和方向向量进行分子对齐，
        通过匹配药效团特征来优化分子的空间排列。
        
        参数
        ----
        mp_ref_and_relaxed : MoleculePair
            包含参考分子和弛豫后分子的分子对对象
            
        返回
        ----
        float
            最优对齐后的药效团相似性评分 (0-1范围)
            
        注意
        ----
        - 使用Tanimoto相似性度量药效团匹配度
        - 考虑药效团类型、位置和方向信息
        - 不使用扩展点，仅基于核心药效团特征
        """
        # 执行基于药效团的分子对齐
        aligned_fit_anchors, aligned_vectors = mp_ref_and_relaxed.align_with_pharm(
            similarity='tanimoto',      # 使用Tanimoto相似性
            extended_points=False,      # 不使用扩展点
            only_extended=False,        # 不仅使用扩展点
            num_repeats=1,              # 对齐重复次数
            trans_init=False,
            use_jax=False
        )
        pharm_similarity = mp_ref_and_relaxed.sim_aligned_pharm
        return float(pharm_similarity)


class UnconditionalEvalPipeline:
    """
    无条件分子生成评估管道类
    
    该类用于评估无条件生成的分子列表，提供全面的分子质量评估指标。
    
    核心功能：
    - 分子有效性验证（生成前后）
    - 分子图一致性检查
    - 分子唯一性统计
    - 分子多样性分析
    - 分子性质计算（SA分数、QED、logP、fsp3等）
    - 应变能和RMSD计算
    
    评估流程：
    1. 对每个生成分子进行ConfEval评估
    2. 收集各种分子性质和指标
    3. 计算整体统计指标
    4. 生成评估报告
    """

    def __init__(self,
                 generated_mols: List[Tuple[np.ndarray, np.ndarray]],
                 solvent: Optional[str] = None):
        """
        初始化无条件分子生成评估管道
        
        该方法设置评估所需的所有属性和数据结构，为后续的分子评估做准备。
        
        参数说明：
        ----------
        generated_mols : List[Tuple[np.ndarray, np.ndarray]]
            生成的分子列表，每个元素为包含原子序数(N,)和坐标(N,3)的元组
        solvent : Optional[str], 默认None
            用于xTB弛豫的隐式溶剂模型名称
            
        属性说明：
        ----------
        generated_mols : 输入的生成分子列表
        smiles/smiles_post_opt : 生成前后的SMILES字符串列表
        molblocks/molblocks_post_opt : 生成前后的分子块列表
        num_generated_mols : 生成分子总数
        num_valid/num_valid_post_opt : 有效分子数量
        num_consistent_graph : 图一致分子数量
        strain_energies/rmsds/SA_scores等 : 各种分子性质数组
        frac_valid/frac_unique等 : 整体统计指标
        
        异常处理：
        ----------
        如果输入的分子列表为空，后续评估将产生空结果
        
        使用示例：
        ----------
        >>> pipeline = UnconditionalEvalPipeline(generated_mols, solvent='water')
        >>> pipeline.evaluate(num_processes=4, verbose=True)
        >>> results = pipeline.to_pandas()
        """
        # 基础数据存储
        self.generated_mols = generated_mols                    # 输入的生成分子列表
        self.smiles = []                                        # 有效分子的SMILES字符串列表
        self.smiles_post_opt = []                              # 弛豫后有效分子的SMILES字符串列表
        self.molblocks = []                                     # 有效分子的分子块列表
        self.molblocks_post_opt = []                           # 弛豫后有效分子的分子块列表
        self.num_generated_mols = len(generated_mols)          # 生成分子总数

        # 溶剂设置
        self.solvent = solvent                                  # xTB弛豫使用的隐式溶剂模型

        # 计数器初始化
        self.num_valid = 0                                      # 有效分子数量计数器
        self.num_valid_post_opt = 0                            # 弛豫后有效分子数量计数器
        self.num_consistent_graph = 0                          # 图一致分子数量计数器

        # 个体分子性质数组（弛豫前）
        self.strain_energies = np.empty(self.num_generated_mols)    # 应变能数组
        self.rmsds = np.empty(self.num_generated_mols)              # RMSD值数组
        self.SA_scores = np.empty(self.num_generated_mols)          # 合成可达性分数数组
        self.logPs = np.empty(self.num_generated_mols)              # 脂水分配系数数组
        self.QEDs = np.empty(self.num_generated_mols)               # 类药性分数数组
        self.fsp3s = np.empty(self.num_generated_mols)              # sp3碳原子比例数组
        self.morgan_fps = []                                        # Morgan指纹列表

        # 个体分子性质数组（弛豫后）
        self.strain_energies_post_opt = np.empty(self.num_generated_mols)  # 弛豫后应变能数组
        self.rmsds_post_opt = np.empty(self.num_generated_mols)            # 弛豫后RMSD值数组
        self.SA_scores_post_opt = np.empty(self.num_generated_mols)        # 弛豫后合成可达性分数数组
        self.logPs_post_opt = np.empty(self.num_generated_mols)            # 弛豫后脂水分配系数数组
        self.QEDs_post_opt = np.empty(self.num_generated_mols)             # 弛豫后类药性分数数组
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)            # 弛豫后sp3碳原子比例数组
        self.morgan_fps_post_opt = []                                      # 弛豫后Morgan指纹列表

        # 整体评估指标（待计算）
        self.frac_valid = None                                  # 有效分子比例
        self.frac_valid_post_opt = None                        # 弛豫后有效分子比例
        self.frac_consistent = None                            # 图一致分子比例
        self.frac_unique = None                                # 唯一分子比例
        self.frac_unique_post_opt = None                       # 弛豫后唯一分子比例
        self.avg_graph_diversity = None                        # 平均图多样性
        self.graph_similarity_matrix = None                    # 图相似性矩阵


    def evaluate(self,
                 num_processes: int = 1,
                 verbose: bool = False
                 ):
        """
        运行无条件分子生成评估管道
        
        该方法对所有生成的分子进行全面评估，包括有效性、一致性、唯一性和多样性分析。
        
        参数说明：
        ----------
        num_processes : int, 默认1
            用于xTB弛豫计算的处理器数量
        verbose : bool, 默认False
            是否显示进度条
            
        评估流程：
        ----------
        1. 遍历所有生成的分子
        2. 对每个分子创建ConfEval实例进行评估
        3. 收集分子指纹、SMILES、分子块等信息
        4. 统计有效性和一致性指标
        5. 计算各种分子性质（弛豫前后）
        6. 计算整体统计指标
        
        注意事项：
        ----------
        - 评估过程可能耗时较长，建议使用多进程加速
        - 无效分子的性质值将设为NaN
        """
        # 设置进度条显示
        if verbose:
            pbar = tqdm(enumerate(self.generated_mols), desc='无条件分子评估',
                        total=self.num_generated_mols)
        else:
            pbar = enumerate(self.generated_mols)
            
        # 遍历所有生成的分子进行评估
        for i, gen_mol in pbar:
            atoms, positions = gen_mol                          # 提取原子序数和坐标
            # 创建构象评估实例
            conf_eval = ConfEval(atoms=atoms, positions=positions, 
                               solvent=self.solvent, num_processes=num_processes)

            # 收集Morgan指纹（如果可用）
            if conf_eval.morgan_fp is not None:
                self.morgan_fps.append(conf_eval.morgan_fp)
                
            # 统计有效分子并收集相关信息
            if conf_eval.is_valid:
                self.num_valid += 1                             # 增加有效分子计数
                self.smiles.append(conf_eval.smiles)            # 收集SMILES字符串
                self.molblocks.append(conf_eval.molblock)       # 收集分子块
                
            # 统计弛豫后有效分子并收集相关信息
            if conf_eval.is_valid_post_opt:
                self.num_valid_post_opt += 1                    # 增加弛豫后有效分子计数
                self.smiles_post_opt.append(conf_eval.smiles_post_opt)      # 收集弛豫后SMILES
                self.molblocks_post_opt.append(conf_eval.molblock_post_opt) # 收集弛豫后分子块

            # 统计图一致性
            self.num_consistent_graph += 1 if conf_eval.is_graph_consistent else 0

            # 收集弛豫前分子性质
            self.strain_energies[i] = self.get_attr(conf_eval, 'strain_energy')    # 应变能
            self.rmsds[i] = self.get_attr(conf_eval, 'rmsd')                       # RMSD值
            self.SA_scores[i] = self.get_attr(conf_eval, 'SA_score')               # 合成可达性分数
            self.QEDs[i] = self.get_attr(conf_eval, 'QED')                         # 类药性分数
            self.logPs[i] = self.get_attr(conf_eval, 'logP')                       # 脂水分配系数
            self.fsp3s[i] = self.get_attr(conf_eval, 'fsp3')                       # sp3碳原子比例

            # 收集弛豫后分子性质
            self.SA_scores_post_opt[i] = self.get_attr(conf_eval, 'SA_score_post_opt')  # 弛豫后合成可达性分数
            self.QEDs_post_opt[i] = self.get_attr(conf_eval, 'QED_post_opt')            # 弛豫后类药性分数
            self.logPs_post_opt[i] = self.get_attr(conf_eval, 'logP_post_opt')          # 弛豫后脂水分配系数
            self.fsp3s_post_opt[i] = self.get_attr(conf_eval, 'fsp3_post_opt')          # 弛豫后sp3碳原子比例

        # 计算整体统计指标
        self.frac_valid = self.get_frac_valid()                                    # 有效分子比例
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()                 # 弛豫后有效分子比例
        self.frac_consistent = self.get_frac_consistent_graph()                   # 图一致分子比例
        self.frac_unique = self.get_frac_unique()                                 # 唯一分子比例
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()               # 弛豫后唯一分子比例
        self.avg_graph_diversity, self.graph_similarity_matrix = self.get_diversity()  # 多样性指标


    def get_attr(self, obj, attr: str):
        """
        安全获取对象属性值
        
        如果属性值为None，则返回np.nan，避免后续计算出错。
        
        参数说明：
        ----------
        obj : object
            目标对象
        attr : str
            属性名称
            
        返回值：
        -------
        属性值或np.nan
        """
        val = getattr(obj, attr)
        if val is None:
            return np.nan
        else:
            return val
    
    def get_frac_valid(self):
        """
        计算有效分子比例
        
        返回值：
        -------
        float : 有效分子数量占总生成分子数量的比例
        """
        return self.num_valid / self.num_generated_mols

    def get_frac_valid_post_opt(self):
        """
        计算弛豫后有效分子比例
        
        返回值：
        -------
        float : 弛豫后有效分子数量占总生成分子数量的比例
        """
        return self.num_valid_post_opt / self.num_generated_mols

    def get_frac_consistent_graph(self):
        """ Fraction of generated molecules that were consistent before and after relaxation. """
        return self.num_consistent_graph / self.num_generated_mols
    
    def get_frac_unique(self):
        """ Fraction of unique smiles extracted pre-optimization in the generated set. """
        if self.num_valid != 0:
            frac = len(set([s for s in self.smiles if s is not None])) / self.num_valid
        else:
            frac = 0.
        return frac

    def get_frac_unique_post_opt(self):
        """ Fraction of unique smiles extracted post-optimization in the generated set. """
        if self.num_valid_post_opt != 0:
            frac = len(set([s for s in self.smiles_post_opt if s is not None])) / self.num_valid_post_opt
        else:
            frac = 0.
        return frac

    def get_diversity(self) -> Tuple[float, np.ndarray]:
        """
        Get average molecular graph diversity (average dissimilarity) as defined by GenBench3D (arXiv:2407.04424)
        and the tanimioto similarity matrix of fingerprints.

        Returns
        -------
        tuple
            avg_diversity : float [0,1] where 1 is more diverse (more dissimilar)
            similarity_matrix : np.ndarray (N,N) similarity matrix
        """
        if self.num_consistent_graph == 0:
            return None, None
        similarity_matrix = np.zeros((self.num_consistent_graph, self.num_consistent_graph))
        running_avg_diversity_sum = 0
        for i, fp1 in enumerate(self.morgan_fps):
            for j, fp2 in enumerate(self.morgan_fps):
                if i == j:
                    similarity_matrix[i,j] = 1
                if i > j: # symmetric
                    similarity_matrix[i,j] = similarity_matrix[j,i]
                else:
                    similarity_matrix[i,j] = TanimotoSimilarity(fp1, fp2)
                    running_avg_diversity_sum += (1 - similarity_matrix[i,j])
        # from GenBench3D: arXiv:2407.04424
        avg_diversity = running_avg_diversity_sum / ((self.num_consistent_graph - 1)*self.num_consistent_graph / 2)
        return avg_diversity, similarity_matrix


    def to_pandas(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Convert the stored attributes to a pd.Series (for global attributes) and pd.DataFrame
        (for attributes relevant to every instance).

        Arguments
        ---------
        self

        Returns
        -------
        Tuple
            pd.Series : global attributes
            pd.DataFrame : attributes for each evaluated sample
        """
        rowwise_attrs = {} # Attributes for each example
        global_attrs = {} # Global attributes

        for key, value in self.__dict__.items():
            if key in ('smiles', 'smiles_post_opt', 'morgan_fps', 'morgan_fps_post_opt'):
                continue
            elif key == 'graph_similarity_matrix' or key == 'graph_similarity_matrix_post_opt':
                global_attrs[key] = value

            elif isinstance(value, (list, tuple, np.ndarray)) and not (isinstance(value, np.ndarray) and value.ndim == 0):
                rowwise_attrs[key] = value
            else:
                global_attrs[key] = value

        print("======== 调试信息：检查各属性数组长度 ========")
        for key, value in rowwise_attrs.items():
            print(f"属性 '{key}' 的长度: {len(value)}")
        print("============================================")

        df_rowwise = pd.DataFrame(rowwise_attrs)
        series_global = pd.Series(global_attrs)

        return series_global, df_rowwise


class ConditionalEvalPipeline:
    """
    条件分子生成评估管道类
    
    该类专门用于评估基于条件生成的分子，通过与参考分子的比较来评估生成分子的质量。
    
    核心功能：
    - 条件一致性评估：评估生成分子与条件约束的一致性
    - 3D相似性评估：计算表面、静电势、药效团相似性
    - 分子性质评估：计算SA评分、QED、logP等分子性质
    - 多维度对齐：支持基于不同条件的分子对齐
    
    评估流程：
    1. 参考分子预处理和特征提取
    2. 生成分子有效性验证
    3. 构象优化和弛豫
    4. 多维度相似性计算
    5. 统计指标汇总
    """

    def __init__(self,
                 ref_molec: Molecule,
                 generated_mols: List[Tuple[np.ndarray, np.ndarray]],
                 condition: str,
                 num_surf_points: int = 400,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 ):
        """
        初始化条件分子生成评估管道
        
        该方法设置评估所需的所有参数和数据结构，包括参考分子处理、
        生成分子存储、评估指标初始化等。
        
        重要假设：
        - 参考分子必须包含用于条件生成的3D表示（表面、静电势或药效团）
        - 生成分子的表面点数必须与参考分子匹配
        - 药效团表示方式必须与生成时保持一致
        
        参数说明：
        ----------
        ref_molec : Molecule
            用于条件生成的参考/目标分子对象，必须包含用于条件生成的3D表示
            （即表面、静电势或药效团信息）
        generated_mols : List[Tuple[np.ndarray, np.ndarray]]
            生成分子列表，每个元素为包含原子序数(N,)和坐标(N,3)的元组
        condition : str
            分子生成时使用的条件类型，可选值：'surface'、'esp'、'pharm'、'all'
            用于确定对齐策略
        num_surf_points : int, 默认=400
            用于相似性评分的表面点采样数量，必须与ref_molec中的表面点数匹配
        pharm_multi_vector : Optional[bool]
            是否使用多向量表示芳香性/氢键受体/氢键供体特征
            必须与联合生成时的设置以及ref_molec的设置保持一致
        solvent : Optional[str]
            用于xTB弛豫的溶剂类型
        
        属性说明：
        ----------
        self.ref_molec : Molecule - 参考分子对象
        self.generated_mols : List - 生成分子列表
        self.condition : str - 条件类型
        self.num_surf_points : int - 表面点数量
        self.lam : float - ESP对齐的最优lambda参数
        self.ref_mol_* : 参考分子的各种性质（SA评分、QED、logP等）
        self.sims_*_upper_bound : 相似性评分的上界
        self.smiles* : SMILES字符串列表（弛豫前后）
        self.molblocks* : 分子块列表（弛豫前后）
        self.num_valid* : 有效分子数量统计
        self.*_scores* : 各种分子性质评分数组
        self.sims_* : 各种相似性评分数组
        
        异常处理：
        ----------
        ValueError : 当参考分子的表面点数与指定的num_surf_points不匹配时抛出
        
        使用示例：
        ----------
        >>> ref_mol = Molecule(...)  # 包含表面信息的参考分子
        >>> gen_mols = [(atoms1, pos1), (atoms2, pos2), ...]  # 生成的分子
        >>> pipeline = ConditionalEvalPipeline(
        ...     ref_molec=ref_mol,
        ...     generated_mols=gen_mols,
        ...     condition='surface',
        ...     num_surf_points=400
        ... )
        >>> pipeline.evaluate(num_processes=4)
        >>> results = pipeline.to_pandas()
        """
        self.generated_mols = generated_mols
        self.num_generated_mols = len(self.generated_mols)
        self.solvent = solvent        

        self.pharm_multi_vector = pharm_multi_vector
        self.condition = condition
        self.num_surf_points = num_surf_points
        self.lam = 0.3 # Optimal lambda for probe_radius=1.2 -> ONLY TO BE USED FOR ESP ALIGNMENT
        self.lam_scaled = self.lam * LAM_SCALING # -> ONLY TO BE USED FOR get_overlap_esp*

        self.ref_molec = ref_molec
        if self.ref_molec.num_surf_points != self.num_surf_points:
            raise ValueError(
                f'The number of surface points in the reference molecule ({self.ref_molec.num_surf_points}) does not match `num_surf_points` ({self.num_surf_points}).'
            )
        self.ref_molblock = Chem.MolToMolBlock(ref_molec.mol)
        self.ref_mol_SA_score = sascorer.calculateScore(Chem.RemoveHs(self.ref_molec.mol))
        self.ref_mol_QED = QED.qed(self.ref_molec.mol)
        self.ref_mol_logP = Crippen.MolLogP(self.ref_molec.mol)
        self.ref_mol_fsp3 = Lipinski.FractionCSP3(self.ref_molec.mol)
        self.ref_mol_morgan_fp = morgan_fp_gen.GetFingerprint(mol=Chem.RemoveHs(self.ref_molec.mol))
        resampling_scores = self.resampling_surf_scores()
        self.ref_surf_resampling_scores = resampling_scores[0]
        self.ref_surf_esp_resampling_scores = resampling_scores[1]
        self.sims_surf_upper_bound = max(self.ref_surf_resampling_scores)
        self.sims_esp_upper_bound = max(self.ref_surf_esp_resampling_scores)

        # 分子表示存储（弛豫前后）
        self.smiles = []  # 弛豫前的SMILES字符串列表
        self.smiles_post_opt = []  # 弛豫后的SMILES字符串列表
        self.molblocks = []  # 弛豫前的分子块列表
        self.molblocks_post_opt = []  # 弛豫后的分子块列表
        
        # 计数器初始化
        self.num_valid = 0  # 有效分子数量（弛豫前）
        self.num_valid_post_opt = 0  # 有效分子数量（弛豫后）
        self.num_consistent_graph = 0  # 图结构一致的分子数量

        # 个体分子性质数组（弛豫前）
        self.strain_energies = np.empty(self.num_generated_mols)  # 应变能数组
        self.rmsds = np.empty(self.num_generated_mols)  # RMSD值数组
        self.SA_scores = np.empty(self.num_generated_mols)  # SA合成可达性评分数组
        self.logPs = np.empty(self.num_generated_mols)  # 脂水分配系数数组
        self.QEDs = np.empty(self.num_generated_mols)  # QED药物相似性评分数组
        self.fsp3s = np.empty(self.num_generated_mols)  # sp3碳原子比例数组
        self.morgan_fps = []  # Morgan指纹列表

        # 个体分子性质数组（弛豫后）
        self.SA_scores_post_opt = np.empty(self.num_generated_mols)  # 弛豫后SA评分数组
        self.logPs_post_opt = np.empty(self.num_generated_mols)  # 弛豫后logP数组
        self.QEDs_post_opt = np.empty(self.num_generated_mols)  # 弛豫后QED评分数组
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)  # 弛豫后sp3比例数组
        self.morgan_fps_post_opt = []  # 弛豫后Morgan指纹列表

        # 整体评估指标
        self.frac_valid = None  # 有效分子比例（弛豫前）
        self.frac_valid_post_opt = None  # 有效分子比例（弛豫后）
        self.frac_consistent = None  # 图结构一致分子比例
        self.frac_unique = None  # 唯一分子比例（弛豫前）
        self.frac_unique_post_opt = None  # 唯一分子比例（弛豫后）
        self.avg_graph_diversity = None  # 平均图多样性
        
        # 3D相似性评分数组
        self.sims_surf_target = np.empty(self.num_generated_mols)  # 表面相似性（原始）
        self.sims_esp_target = np.empty(self.num_generated_mols)  # 静电势相似性（原始）
        self.sims_pharm_target = np.empty(self.num_generated_mols)  # 药效团相似性（原始）

        self.sims_surf_target_relax = np.empty(self.num_generated_mols)  # 表面相似性（弛豫后）
        self.sims_esp_target_relax = np.empty(self.num_generated_mols)  # 静电势相似性（弛豫后）
        self.sims_pharm_target_relax = np.empty(self.num_generated_mols)  # 药效团相似性（弛豫后）

        self.sims_surf_target_relax_optimal = np.empty(self.num_generated_mols)  # 表面相似性（最优对齐）
        self.sims_esp_target_relax_optimal = np.empty(self.num_generated_mols)  # 静电势相似性（最优对齐）
        self.sims_pharm_target_relax_optimal = np.empty(self.num_generated_mols)  # 药效团相似性（最优对齐）

        self.sims_surf_target_relax_esp_aligned = np.empty(self.num_generated_mols)  # 表面相似性（ESP对齐）
        self.sims_pharm_target_relax_esp_aligned = np.empty(self.num_generated_mols)  # 药效团相似性（ESP对齐）
        
        # 2D相似性评分数组
        self.graph_similarities = np.empty(self.num_generated_mols)  # 图相似性（弛豫前）
        self.graph_similarities_post_opt = np.empty(self.num_generated_mols)  # 图相似性（弛豫后）


    def evaluate(self,
                 num_processes: int = 1,
                 verbose: bool=False):
        """ 
        Run conditional evaluation on every generated molecule and store collective values.

        Arguments
        ---------
        num_processes : int number of processors to use for xtb relaxation
        verbose : bool for whether to display tqdm

        Returns
        -------
        None : Just updates the class attributes
        """
        if verbose:
            pbar = tqdm(enumerate(self.generated_mols),
                        desc='Conditional Eval',
                        total=self.num_generated_mols)
        else:
            pbar = enumerate(self.generated_mols)
        for i, gen_mol in pbar:
            atoms, positions = gen_mol

            cond_eval = ConditionalEval(
                ref_molec=self.ref_molec,
                atoms=atoms,
                positions=positions,
                condition=self.condition,
                num_surf_points=self.num_surf_points,
                pharm_multi_vector=self.pharm_multi_vector,
                num_processes=num_processes,
                solvent=self.solvent
            )

            # Conformer attributes
            self.num_consistent_graph += 1 if cond_eval.is_graph_consistent else 0
            self.molblocks.append(cond_eval.molblock)
            self.molblocks_post_opt.append(cond_eval.molblock_post_opt)

            if cond_eval.is_valid:
                self.num_valid += 1
                self.smiles.append(cond_eval.smiles)
            else:
                self.smiles.append(None)
            if cond_eval.is_valid_post_opt:
                self.num_valid_post_opt += 1
                self.smiles_post_opt.append(cond_eval.smiles_post_opt)
            else:
                self.smiles_post_opt.append(None)

            self.strain_energies[i] = self.get_attr(cond_eval, 'strain_energy')
            self.rmsds[i] = self.get_attr(cond_eval, 'rmsd')
            self.SA_scores[i] = self.get_attr(cond_eval, 'SA_score')
            self.QEDs[i] = self.get_attr(cond_eval, 'QED')
            self.logPs[i] = self.get_attr(cond_eval, 'logP')
            self.fsp3s[i] = self.get_attr(cond_eval, 'fsp3')
            if cond_eval.morgan_fp is not None:
                self.graph_similarities[i] = TanimotoSimilarity(cond_eval.morgan_fp, self.ref_mol_morgan_fp)
            else:
                self.graph_similarities[i] = np.nan
            if cond_eval.morgan_fp_post_opt is not None:
                self.graph_similarities_post_opt[i] = TanimotoSimilarity(cond_eval.morgan_fp_post_opt, self.ref_mol_morgan_fp)
            else:
                self.graph_similarities_post_opt[i] = np.nan
            
            self.SA_scores_post_opt[i] = self.get_attr(cond_eval, 'SA_score_post_opt')
            self.QEDs_post_opt[i] = self.get_attr(cond_eval, 'QED_post_opt')
            self.logPs_post_opt[i] = self.get_attr(cond_eval, 'logP_post_opt')
            self.fsp3s_post_opt[i] = self.get_attr(cond_eval, 'fsp3_post_opt')

            # Conditional attributes
            self.sims_surf_target[i] = self.get_attr(cond_eval, 'sim_surf_target')
            self.sims_esp_target[i] = self.get_attr(cond_eval, 'sim_esp_target')
            self.sims_pharm_target[i] = self.get_attr(cond_eval, 'sim_pharm_target')

            self.sims_surf_target_relax[i] = self.get_attr(cond_eval, 'sim_surf_target_relax')
            self.sims_esp_target_relax[i] = self.get_attr(cond_eval, 'sim_esp_target_relax')
            self.sims_pharm_target_relax[i] = self.get_attr(cond_eval, 'sim_pharm_target_relax')      

            self.sims_surf_target_relax_optimal[i] = self.get_attr(cond_eval, 'sim_surf_target_relax_optimal')
            self.sims_esp_target_relax_optimal[i] = self.get_attr(cond_eval, 'sim_esp_target_relax_optimal')
            self.sims_pharm_target_relax_optimal[i] = self.get_attr(cond_eval, 'sim_pharm_target_relax_optimal')

            self.sims_surf_target_relax_esp_aligned[i] = self.get_attr(cond_eval, 'sim_surf_target_relax_esp_aligned')
            self.sims_pharm_target_relax_esp_aligned[i] = self.get_attr(cond_eval, 'sim_pharm_target_relax_esp_aligned')

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity = self.get_diversity()

    
    def resampling_surf_scores(self) -> Union[np.ndarray, None]:
        """
        Capture distribution of surface similarity and surface ESP scores caused by resampling
        surface.

        Returns
        -------
        Tuple
            surf_scores : np.ndarray or None (if not relevant)
            esp_scores : np.ndarray or None (if not relevant)
        """
        surf_scores = np.empty(50)
        esp_scores = np.empty(50)
        for i in range(50):
            molec = Molecule(mol=self.ref_molec.mol,
                             num_surf_points=self.num_surf_points,
                             probe_radius=self.ref_molec.probe_radius,
                             partial_charges=np.array(self.ref_molec.partial_charges))
            surf_scores[i] = get_overlap_np(
                self.ref_molec.surf_pos,
                molec.surf_pos,
                alpha=ALPHA(molec.num_surf_points)
            )
            esp_scores[i] = get_overlap_esp_np(
                centers_1=self.ref_molec.surf_pos, 
                centers_2=molec.surf_pos,
                charges_1=self.ref_molec.surf_esp,
                charges_2=molec.surf_esp,
                alpha=ALPHA(molec.num_surf_points),
                lam=self.lam_scaled
            )
            
        return surf_scores, esp_scores
            

    def get_attr(self, obj, attr: str):
        """ Gets an attribute of `obj` via the string name. If it is None, then return np.nan """
        val = getattr(obj, attr)
        if val is None:
            return np.nan
        else:
            return val
        
    def get_frac_valid(self):
        """ Fraction of generated molecules that were valid. """
        return self.num_valid / self.num_generated_mols

    def get_frac_valid_post_opt(self):
        """ Fraction of generated molecules that were valid after relaxation. """
        return self.num_valid_post_opt / self.num_generated_mols

    def get_frac_consistent_graph(self):
        """ Fraction of generated molecules that were consistent before and after relaxation. """
        return self.num_consistent_graph / self.num_generated_mols
    
    def get_frac_unique(self):
        """ Fraction of unique smiles extracted pre-optimization in the generated set. """
        if self.num_valid != 0:
            frac = len(set([s for s in self.smiles if s is not None])) / self.num_valid
        else:
            frac = 0.
        return frac

    def get_frac_unique_post_opt(self):
        """ Fraction of unique smiles extracted post-optimization in the generated set. """
        if self.num_valid_post_opt != 0:
            frac = len(set([s for s in self.smiles_post_opt if s is not None])) / self.num_valid_post_opt
        else:
            frac = 0.
        return frac


    def get_diversity(self) -> float:
        """
        Get average molecular graph diversity (average dissimilarity) w.r.t. target.

        Returns
        -------
        avg_diversity : float [0,1] where 1 is more diverse (more dissimilar)
        """
        avg_diversity = np.nanmean(1 - self.graph_similarities)
        return avg_diversity
    
    
    def to_pandas(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Convert the stored attributes to a pd.Series (for global attributes) and pd.DataFrame
        (for attributes relevant to every instance).

        Arguments
        ---------
        self

        Returns
        -------
        Tuple
            pd.Series : global attributes
            pd.DataFrame : attributes for each evaluated sample
        """
        rowwise_attrs = {} # Attributes for each example
        global_attrs = {} # Global attributes

        for key, value in self.__dict__.items():
            if key in ('smiles', 'smiles_post_opt', 'morgan_fps', 'morgan_fps_post_opt', 'ref_molec'):
                continue
            elif key in ('ref_surf_resampling_scores', 'ref_surf_esp_resampling_scores'):
                global_attrs[key] = value

            elif isinstance(value, (list, tuple, np.ndarray)) and not (isinstance(value, np.ndarray) and value.ndim == 0):
                rowwise_attrs[key] = value
            else:
                global_attrs[key] = value

        df_rowwise = pd.DataFrame(rowwise_attrs)
        series_global = pd.Series(global_attrs)

        return series_global, df_rowwise


def resample_surf_scores(ref_molec: Molecule,
                         num_samples: int = 20,
                         eval_surf: bool = True,
                         eval_esp: bool = True,
                         lam_scaled: float = 0.3 * LAM_SCALING
                         ) -> Tuple[Union[np.ndarray, None]]:
    """
    Helper function to get a baseline of resampling the surface and scoring.
    """
    surf_scores = np.empty(num_samples)
    esp_scores = np.empty(num_samples)
    if eval_surf is None or ref_molec.num_surf_points is None:
        return None, None
    if eval_esp is None:
        esp_scores = None
    for i in range(num_samples):
        molec = Molecule(mol=ref_molec.mol,
                         num_surf_points=ref_molec.num_surf_points,
                         probe_radius=ref_molec.probe_radius,
                         partial_charges=np.array(ref_molec.partial_charges))
        surf_scores[i] = get_overlap_np(ref_molec.surf_pos,
                                        molec.surf_pos,
                                        alpha=ALPHA(molec.num_surf_points))
        if eval_esp:
            esp_scores[i] = get_overlap_esp_np(centers_1=ref_molec.surf_pos, 
                                               centers_2=molec.surf_pos,
                                               charges_1=ref_molec.surf_esp,
                                               charges_2=molec.surf_esp,
                                               alpha=ALPHA(molec.num_surf_points),
                                               lam=lam_scaled)
    return surf_scores, esp_scores


class ConsistencyEvalPipeline:
    """
    一致性评估管道类
    
    用于评估联合生成的分子及其多模态特征（表面点、静电势、药效团）之间的一致性。
    该类专门处理多模态分子生成模型的评估，验证生成的不同模态特征是否相互一致。
    
    核心功能：
    - 评估生成分子的化学合理性和有效性
    - 计算多模态特征之间的一致性评分
    - 提供上界和下界基准评分
    - 支持弛豫前后的一致性比较
    
    评估流程：
    1. 验证生成分子的化学结构合理性
    2. 计算分子性质（SA评分、logP、QED等）
    3. 评估多模态特征一致性（表面、静电势、药效团）
    4. 计算统计指标（有效性、唯一性、多样性）
    """

    def __init__(self,
                 generated_mols: List[Tuple[np.ndarray, np.ndarray]],
                 generated_surf_points: Optional[List[np.ndarray]] = None,
                 generated_surf_esp: Optional[List[np.ndarray]] = None,
                 generated_pharm_feats: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
                 probe_radius: float = 1.2,
                 pharm_multi_vector: Optional[bool] = None,
                 solvent: Optional[str] = None,
                 random_molblock_charges: Optional[List[Tuple]] = None
                 ):
        """
        初始化一致性评估管道
        
        用于设置评估所需的所有参数和数据结构，包括生成的分子、多模态特征
        以及评估配置参数。
        
        参数说明：
        ---------
        generated_mols : List[Tuple[np.ndarray, np.ndarray]]
            生成的分子列表，每个元素为包含原子序数(N,)和坐标(N,3)的元组
            
        generated_surf_points : Optional[List[np.ndarray]]
            生成的表面点云列表，每个数组形状为(M, 3)，表示表面点的3D坐标
            
        generated_surf_esp : Optional[List[np.ndarray]]
            生成的静电势值列表，每个数组形状为(M,)，对应表面点的静电势
            
        generated_pharm_feats : Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]
            生成的药效团特征列表，每个元组包含：
            - 药效团类型数组(P,)：整数表示的药效团类型
            - 锚点坐标数组(P, 3)：药效团锚点的3D坐标
            - 方向向量数组(P, 3)：药效团的单位方向向量
            
        probe_radius : float, 默认1.2
            溶剂可及表面计算的探针半径（埃）
            
        pharm_multi_vector : Optional[bool]
            是否使用多向量表示芳香环/氢键受体/氢键供体药效团
            需与联合生成时的设置保持一致
            
        solvent : Optional[str]
            xTB弛豫计算使用的溶剂类型
            
        random_molblock_charges : Optional[List[Tuple]]
            随机分子块电荷列表，用于随机选择并与生成样本对齐
        
        属性说明：
        ---------
        初始化后会创建以下主要属性：
        - 基础数据：生成分子数量、溶剂设置、探针半径等
        - 分子表示：SMILES字符串、分子块等
        - 计数器：有效分子数、一致性分子数等
        - 性质数组：SA评分、logP、QED、Fsp3等
        - 相似性评分：表面、静电势、药效团一致性评分
        - 统计指标：有效性、唯一性、多样性等
        
        异常处理：
        ---------
        - 当提供静电势但未提供表面点时抛出ValueError
        - 自动验证各模态特征列表长度一致性
        
        使用示例：
        ---------
        >>> pipeline = ConsistencyEvalPipeline(
        ...     generated_mols=[(atoms1, pos1), (atoms2, pos2)],
        ...     generated_surf_points=[surf1, surf2],
        ...     generated_surf_esp=[esp1, esp2],
        ...     solvent='water'
        ... )
        >>> pipeline.evaluate(num_processes=4)
        """
        # 基础数据存储
        self.generated_mols = generated_mols  # 生成的分子列表
        self.num_generated_mols = len(self.generated_mols)  # 生成分子总数
        self.solvent = solvent  # 溶剂类型
        self.probe_radius = probe_radius  # 探针半径
        self.random_molblock_charges = random_molblock_charges  # 随机分子块电荷
        if self.random_molblock_charges is not None:
            self.num_random_molblock_charges = len(self.random_molblock_charges)  # 随机分子块数量
        else:
            self.num_random_molblock_charges = None

        # 多模态特征验证和存储
        # 检查表面点云数据长度一致性
        if generated_surf_points is not None:
            assert self.num_generated_mols == len(generated_surf_points)
        self.generated_surf_points = generated_surf_points  # 生成的表面点云
        
        # 检查静电势数据长度一致性
        if generated_surf_esp is not None:
            assert self.num_generated_mols == len(generated_surf_esp)
        self.generated_surf_esp = generated_surf_esp  # 生成的静电势值
        
        # 验证静电势和表面点的依赖关系
        if self.generated_surf_esp is not None and self.generated_surf_points is None:
            raise ValueError(f'`generated_surf_pos` must also be provided if `generated_surf_esp` is given.')

        # 药效团特征处理
        if generated_pharm_feats is not None:  # 解包药效团特征
            self.generated_pharm_feats = generated_pharm_feats
        else:
            self.generated_pharm_feats = None

        self.pharm_multi_vector = pharm_multi_vector  # 药效团多向量表示设置

        # 分子表示存储
        self.smiles = []  # 弛豫前SMILES字符串列表
        self.smiles_post_opt = []  # 弛豫后SMILES字符串列表
        self.molblocks = []  # 弛豫前分子块列表
        self.molblocks_post_opt = []  # 弛豫后分子块列表
        
        # 计数器初始化
        self.num_valid = 0  # 有效分子数量
        self.num_valid_post_opt = 0  # 弛豫后有效分子数量
        self.num_consistent_graph = 0  # 图结构一致的分子数量

        # 个体分子性质数组（弛豫前）
        self.strain_energies = np.empty(self.num_generated_mols)  # 应变能数组
        self.rmsds = np.empty(self.num_generated_mols)  # RMSD值数组
        self.SA_scores = np.empty(self.num_generated_mols)  # 合成可及性评分数组
        self.logPs = np.empty(self.num_generated_mols)  # 脂水分配系数数组
        self.QEDs = np.empty(self.num_generated_mols)  # 类药性评分数组
        self.fsp3s = np.empty(self.num_generated_mols)  # sp3碳原子比例数组
        self.morgan_fps = []  # Morgan指纹列表

        # 个体分子性质数组（弛豫后）
        self.SA_scores_post_opt = np.empty(self.num_generated_mols)  # 弛豫后合成可及性评分
        self.logPs_post_opt = np.empty(self.num_generated_mols)  # 弛豫后脂水分配系数
        self.QEDs_post_opt = np.empty(self.num_generated_mols)  # 弛豫后类药性评分
        self.fsp3s_post_opt = np.empty(self.num_generated_mols)  # 弛豫后sp3碳原子比例
        self.morgan_fps_post_opt = []  # 弛豫后Morgan指纹列表

        # 整体评估指标
        self.frac_valid = None  # 有效分子比例
        self.frac_valid_post_opt = None  # 弛豫后有效分子比例
        self.frac_consistent = None  # 一致性分子比例
        self.frac_unique = None  # 唯一分子比例
        self.frac_unique_post_opt = None  # 弛豫后唯一分子比例
        self.avg_graph_diversity = None  # 平均图多样性
        self.graph_similarity_matrix = None  # 图相似性矩阵
        self.avg_graph_diversity_post_opt = None  # 弛豫后平均图多样性
        self.graph_similarity_matrix_post_opt = None  # 弛豫后图相似性矩阵
        
        # 3D相似性评分数组（一致性评估）
        self.sims_surf_consistent = np.empty(self.num_generated_mols)  # 表面一致性评分
        self.sims_esp_consistent = np.empty(self.num_generated_mols)  # 静电势一致性评分
        self.sims_pharm_consistent = np.empty(self.num_generated_mols)  # 药效团一致性评分

        # 上界基准评分数组
        self.sims_surf_upper_bound = np.empty(self.num_generated_mols)  # 表面评分上界
        self.sims_esp_upper_bound = np.empty(self.num_generated_mols)  # 静电势评分上界

        # 下界基准评分数组
        self.sims_surf_lower_bound = np.empty(self.num_generated_mols)  # 表面评分下界
        self.sims_esp_lower_bound = np.empty(self.num_generated_mols)  # 静电势评分下界
        self.sims_pharm_lower_bound = np.empty(self.num_generated_mols)  # 药效团评分下界

        # 弛豫后一致性评分数组
        self.sims_surf_consistent_relax = np.empty(self.num_generated_mols)  # 弛豫后表面一致性
        self.sims_esp_consistent_relax = np.empty(self.num_generated_mols)  # 弛豫后静电势一致性
        self.sims_pharm_consistent_relax = np.empty(self.num_generated_mols)  # 弛豫后药效团一致性

        # 最优对齐弛豫后一致性评分数组
        self.sims_surf_consistent_relax_optimal = np.empty(self.num_generated_mols)  # 最优对齐表面一致性
        self.sims_esp_consistent_relax_optimal = np.empty(self.num_generated_mols)  # 最优对齐静电势一致性
        self.sims_pharm_consistent_relax_optimal = np.empty(self.num_generated_mols)  # 最优对齐药效团一致性


    def evaluate(self,
                 num_processes: int = 1,
                 verbose: bool=False):
        """ 
        执行一致性评估
        
        对每个生成的分子进行一致性评估，并存储集体评估结果。
        该方法会逐个处理生成的分子，计算其与参考数据的一致性评分。
        
        参数:
            num_processes (int, 可选): 用于xTB弛豫计算的处理器数量，默认为1
            verbose (bool, 可选): 是否显示进度条，默认为False
        
        返回:
            None: 该方法不返回值，仅更新类的属性
        
        功能说明:
            - 遍历所有生成的分子进行一致性评估
            - 计算分子的有效性、唯一性和图一致性
            - 评估表面、静电势和药效团的一致性
            - 进行xTB弛豫优化并计算弛豫后的一致性
            - 更新类的各种评估指标属性
        
        注意:
            - 如果提供了随机分子块电荷数据，会随机选择用于下界计算
            - 所有评估结果都存储在类的属性中，可通过相应的getter方法获取
        """

        if verbose:
            pbar = tqdm(enumerate(self.generated_mols), desc='Consistency Eval',
                        total=self.num_generated_mols)
        else:
            pbar = enumerate(self.generated_mols)
        for i, gen_mol in pbar:
            atoms, positions = gen_mol
            surf_points = self.generated_surf_points[i] if self.generated_surf_points is not None else None
            surf_esp = self.generated_surf_esp[i] if self.generated_surf_esp is not None else None
            pharm_feats = self.generated_pharm_feats[i] if self.generated_pharm_feats is not None else None
            if self.num_random_molblock_charges is not None:
                rand_ind_for_lower_bound = RNG.choice(self.num_random_molblock_charges, 1)[0]
            else:
                rand_ind_for_lower_bound = 0

            consis_eval = ConsistencyEval(
                atoms=atoms,
                positions=positions,
                surf_points=surf_points,
                surf_esp=surf_esp,
                pharm_feats=pharm_feats,
                pharm_multi_vector=self.pharm_multi_vector,
                probe_radius=self.probe_radius,
                num_processes=num_processes,
                solvent=self.solvent
            )

            # Conformer attributes
            self.num_consistent_graph += 1 if consis_eval.is_graph_consistent else 0

            self.molblocks.append(consis_eval.molblock)
            self.molblocks_post_opt.append(consis_eval.molblock_post_opt)
            self.smiles.append(consis_eval.smiles)

            if consis_eval.is_valid:
                self.num_valid += 1

                # Compute similarity score lower bounds
                if self.num_random_molblock_charges is not None:
                    rand_molblock_charges = self.random_molblock_charges[rand_ind_for_lower_bound]
                    rand_molec = Molecule(
                        mol=Chem.MolFromMolBlock(rand_molblock_charges[0], removeHs=False),
                        num_surf_points=consis_eval.molec_regen.num_surf_points,
                        partial_charges=np.array(rand_molblock_charges[1]),
                        pharm_multi_vector=consis_eval.molec_regen.pharm_multi_vector
                    )

                    mp = MoleculePair(ref_mol=consis_eval.molec_regen,
                                      fit_mol=rand_molec,
                                      num_surf_points=consis_eval.molec_regen.num_surf_points)

                    # align and compare to molec_regen
                    if consis_eval.molec_regen.surf_pos is not None:
                        mp.align_with_surf(alpha=ALPHA(mp.num_surf_points),
                                           num_repeats=50,
                                           trans_init=False,
                                           use_jax=False)
                        self.sims_surf_lower_bound[i] = mp.sim_aligned_surf
                    else:
                        self.sims_surf_lower_bound[i] = np.nan
                    if consis_eval.molec_regen.surf_esp is not None:
                        mp.align_with_esp(alpha=ALPHA(mp.num_surf_points),
                                          lam=consis_eval.lam_scaled,
                                          num_repeats=50,
                                          trans_init=False,
                                          use_jax=False)
                        self.sims_esp_lower_bound[i] = mp.sim_aligned_esp
                    else:
                        self.sims_esp_lower_bound[i] = np.nan
                    if consis_eval.molec_regen.pharm_ancs is not None:
                        mp.align_with_pharm(num_repeats=50,
                                            trans_init=False,
                                            use_jax=False)
                        self.sims_pharm_lower_bound[i] = mp.sim_aligned_pharm
                    else:
                        self.sims_pharm_lower_bound[i] = np.nan
                else:
                    self.sims_surf_lower_bound[i] = np.nan
                    self.sims_esp_lower_bound[i] = np.nan
                    self.sims_pharm_lower_bound[i] = np.nan

            if consis_eval.is_valid_post_opt:
                self.num_valid_post_opt += 1
                self.smiles_post_opt.append(consis_eval.smiles_post_opt)

            # only compute upper bound if consistent
            if consis_eval.is_valid and consis_eval.is_valid_post_opt:
                # Upper bound
                surf_scores, esp_scores = self.resampling_upper_bounds(
                    consis_eval=consis_eval,
                    num_samples=5
                )
                if surf_scores is not None:
                    self.sims_surf_upper_bound[i] = surf_scores
                else:
                    self.sims_surf_upper_bound[i] = np.nan

                if esp_scores is not None:
                    self.sims_esp_upper_bound[i] = esp_scores
                else:
                    self.sims_esp_upper_bound[i] = np.nan
            else:
                self.sims_esp_upper_bound[i] = np.nan
                self.sims_surf_upper_bound[i] = np.nan

            self.strain_energies[i] = self.get_attr(consis_eval, 'strain_energy')
            self.rmsds[i] = self.get_attr(consis_eval, 'rmsd')
            self.SA_scores[i] = self.get_attr(consis_eval, 'SA_score')
            self.QEDs[i] = self.get_attr(consis_eval, 'QED')
            self.logPs[i] = self.get_attr(consis_eval, 'logP')
            self.fsp3s[i] = self.get_attr(consis_eval, 'fsp3')

            self.SA_scores_post_opt[i] = self.get_attr(consis_eval, 'SA_score_post_opt')
            self.QEDs_post_opt[i] = self.get_attr(consis_eval, 'QED_post_opt')
            self.logPs_post_opt[i] = self.get_attr(consis_eval, 'logP_post_opt')
            self.fsp3s_post_opt[i] = self.get_attr(consis_eval, 'fsp3_post_opt')

            # Conditional attributes
            self.sims_surf_consistent[i] = self.get_attr(consis_eval, 'sim_surf_consistent')
            self.sims_esp_consistent[i] = self.get_attr(consis_eval, 'sim_esp_consistent')
            self.sims_pharm_consistent[i] = self.get_attr(consis_eval, 'sim_pharm_consistent')

            self.sims_surf_consistent_relax[i] = self.get_attr(consis_eval, 'sim_surf_consistent_relax')
            self.sims_esp_consistent_relax[i] = self.get_attr(consis_eval, 'sim_esp_consistent_relax')
            self.sims_pharm_consistent_relax[i] = self.get_attr(consis_eval, 'sim_pharm_consistent_relax')

            self.sims_surf_consistent_relax_optimal[i] = self.get_attr(consis_eval, 'sim_surf_consistent_relax_optimal')
            self.sims_esp_consistent_relax_optimal[i] = self.get_attr(consis_eval, 'sim_esp_consistent_relax_optimal')
            self.sims_pharm_consistent_relax_optimal[i] = self.get_attr(consis_eval, 'sim_pharm_consistent_relax_optimal')

        self.frac_valid = self.get_frac_valid()
        self.frac_valid_post_opt = self.get_frac_valid_post_opt()
        self.frac_consistent = self.get_frac_consistent_graph()
        self.frac_unique = self.get_frac_unique()
        self.frac_unique_post_opt = self.get_frac_unique_post_opt()
        self.avg_graph_diversity, self.graph_similarity_matrix = self.get_diversity(post_opt=False)
        self.avg_graph_diversity_post_opt, self.graph_similarity_matrix_post_opt = self.get_diversity(post_opt=True)


    def resampling_surf_scores(self,
                               consis_eval: ConsistencyEval,
                               num_samples: int = 20) -> Tuple[Union[np.ndarray, None]]:
        """
        Capture distribution of surface similarity and surface ESP scores caused by resampling
        surface.
        
        Arguments
        ---------
        consis_eval : ConsistencyEval obj to check similarity scores caused by resampling
        num_samples : int (default = 20) number of times to resample surface and score

        Returns
        -------
        Tuple
            surf_scores : np.ndarray or None (if not relevant)
            esp_scores : np.ndarray or None (if not relevant)
        """
        ref_molec = consis_eval.molec
        surf_scores, esp_scores = resample_surf_scores(
            ref_molec=ref_molec,
            num_samples=num_samples,
            eval_surf=consis_eval.molec.surf_pos is not None,
            eval_esp=consis_eval.molec.surf_esp is not None,
            lam_scaled=consis_eval.lam_scaled
        )            
        return surf_scores, esp_scores

    
    @staticmethod
    def resampling_upper_bounds(consis_eval: ConsistencyEval,
                                num_samples: int = 5,
                                num_surf_points: Optional[int] = None
                                ) -> Tuple[Union[float, None]]:
        """
        Compute the expectation (upper bound) of similarity score caused by stochastic surface
        sampling by calculating the mean similarity between pairwise comparisons.

        Arguments
        ---------
        consis_eval : ConsistencyEval
        num_samples = 5

        Returns
        -------
        Tuple
            upper_bound_surf : float or None surface similarity upper bound
            upper_bound_esp : float or None ESP similarity upper bound
        """
        eval_surf = consis_eval.molec_post_opt.surf_pos is not None
        eval_esp = consis_eval.molec_post_opt.surf_esp is not None and consis_eval.molec_post_opt.surf_pos is not None
        if eval_surf is False and eval_esp is False:
            return None, None
        
        if num_surf_points is None:
            num_surf_points = consis_eval.num_surf_points

        # extract multiple instances of the interaction profiles
        molecs_ls = []
        for _ in range(num_samples):
            molec_extract = Molecule(
                mol=consis_eval.mol_post_opt,
                num_surf_points=num_surf_points,
                probe_radius=consis_eval.probe_radius,
                partial_charges=consis_eval.partial_charges_post_opt,
            )
            molecs_ls.append(molec_extract)

        # Score all combinations
        all_surf_scores = []
        all_esp_scores = []
        inds_all_combos = list(itertools.combinations(list(range(len(molecs_ls))), 2))

        for inds in inds_all_combos:
            molec_1 = molecs_ls[inds[0]]
            molec_2 = molecs_ls[inds[1]]

            if eval_surf:
                # surface scoring
                score = get_overlap_np(
                    centers_1=molec_1.surf_pos,
                    centers_2=molec_2.surf_pos,
                    alpha=ALPHA(num_surf_points)
                )
                all_surf_scores.append(score)
            else:
                all_surf_scores = None

            if eval_esp:
                # ESP surface scoring
                # MAKE SURE TO SCALE LAMBDA
                score = get_overlap_esp_np(
                    centers_1=molec_1.surf_pos,
                    centers_2=molec_2.surf_pos,
                    charges_1=molec_1.surf_esp,
                    charges_2=molec_2.surf_esp,
                    alpha=ALPHA(num_surf_points),
                    lam = consis_eval.lam_scaled
                )
                all_esp_scores.append(score)
            else:
                all_esp_scores = None

        upper_bound_surf = None
        upper_bound_esp = None
        if all_surf_scores is not None:
            upper_bound_surf = np.nanmean(np.array(all_surf_scores))
        if all_esp_scores is not None:
            upper_bound_esp = np.nanmean(np.array(all_esp_scores))

        return float(upper_bound_surf), float(upper_bound_esp)
            

    def get_attr(self, obj, attr: str):
        """ Gets an attribute of `obj` via the string name. If it is None, then return np.nan """
        val = getattr(obj, attr)
        if val is None:
            return np.nan
        else:
            return val
        
    def get_frac_valid(self):
        """ Fraction of generated molecules that were valid. """
        return self.num_valid / self.num_generated_mols

    def get_frac_valid_post_opt(self):
        """ Fraction of generated molecules that were valid after relaxation. """
        return self.num_valid_post_opt / self.num_generated_mols

    def get_frac_consistent_graph(self):
        """ Fraction of generated molecules that were consistent before and after relaxation. """
        return self.num_consistent_graph / self.num_generated_mols
    
    def get_frac_unique(self):
        """ Fraction of unique smiles extracted pre-optimization in the generated set. """
        if self.num_valid != 0:
            frac = len(set([s for s in self.smiles if s is not None])) / self.num_valid
        else:
            frac = 0.
        return frac

    def get_frac_unique_post_opt(self):
        """ Fraction of unique smiles extracted post-optimization in the generated set. """
        if self.num_valid_post_opt != 0:
            frac = len(set([s for s in self.smiles_post_opt if s is not None])) / self.num_valid_post_opt
        else:
            frac = 0.
        return frac


    def get_diversity(self, post_opt=False) -> Tuple[float, np.ndarray]:
        """
        Get average molecular graph diversity (average dissimilarity) as defined by GenBench3D (arXiv:2407.04424)
        and the tanimioto similarity matrix of fingerprints.

        Returns
        -------
        tuple
            avg_diversity : float [0,1] where 1 is more diverse (more dissimilar)
            similarity_matrix : np.ndarray (N,N) similarity matrix
        """
        if self.num_consistent_graph == 0:
            return None, None
        if post_opt:
            fps = self.morgan_fps
        else:
            fps = self.morgan_fps_post_opt
        similarity_matrix = np.zeros((self.num_consistent_graph, self.num_consistent_graph))
        running_avg_diversity_sum = 0
        for i, fp1 in enumerate(fps):
            for j, fp2 in enumerate(fps):
                if i == j:
                    similarity_matrix[i,j] = 1
                if i > j: # symmetric
                    similarity_matrix[i,j] = similarity_matrix[j,i]
                else:
                    similarity_matrix[i,j] = TanimotoSimilarity(fp1, fp2)
                    running_avg_diversity_sum += (1 - similarity_matrix[i,j])
        # from GenBench3D: arXiv:2407.04424
        avg_diversity = running_avg_diversity_sum / ((self.num_consistent_graph - 1)*self.num_consistent_graph / 2)
        return avg_diversity, similarity_matrix
    

    def to_pandas(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Convert the stored attributes to a pd.Series (for global attributes) and pd.DataFrame
        (for attributes relevant to every instance).

        Arguments
        ---------
        self

        Returns
        -------
        Tuple
            pd.Series : global attributes
            pd.DataFrame : attributes for each evaluated sample
        """
        rowwise_attrs = {} # Attributes for each example
        global_attrs = {} # Global attributes

        for key, value in self.__dict__.items():
            if key in ('random_molblock_charges', 'num_random_molblock_charges', 'smiles',
                       'smiles_post_opt', 'morgan_fps', 'morgan_fps_post_opt'):
                continue
            elif key == 'graph_similarity_matrix' or key == 'graph_similarity_matrix_post_opt':
                global_attrs[key] = value

            elif isinstance(value, (list, tuple, np.ndarray)) and not (isinstance(value, np.ndarray) and value.ndim == 0):
                rowwise_attrs[key] = value
            else:
                global_attrs[key] = value

        df_rowwise = pd.DataFrame(rowwise_attrs)
        series_global = pd.Series(global_attrs)

        return series_global, df_rowwise
