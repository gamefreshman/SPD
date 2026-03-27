"""
DPO训练所需的核心工具类：
1. ShepherdScorer - Shepherd Score评分器
2. OnlineSampler - 在线采样器
3. PreferencePairBuilder - 偏好对构建器
"""

import torch
import numpy as np
import rdkit
from rdkit import Chem
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import pickle

# 采样所需的工具函数
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.inference import inference_sample
# from shepherd.extract import create_rdkit_molecule
from shepherd.extract_shepherd import create_rdkit_molecule

class ShepherdScorer:
    """
    Shepherd Score 评分器
    根据多个指标综合评估分子质量
    """
    
    def __init__(self, params):
        self.params = params
        
        # 强制使用完整的Shepherd Score评分（shepherd_score_utils在src/shepherd/下）
        try:
            from shepherd_score.evaluations.evaluate import ConfEval, ConditionalEvalPipeline
            from shepherd_score.container import Molecule
            self.use_full_score = True
            print("✅ 已加载完整Shepherd Score评分系统（shepherd_score）")
        except ImportError as e:
            # shepherd_score包可能未安装，但不使用简化评分
            raise ImportError(
                f"无法导入shepherd_score包: {e}\n"
                "请安装: cd /home1/zhh/workspace/SPD/src && uv pip install -e ./score"
            )
        
        # 评分权重配置
        self.weights = {
            'rmsd': -1.0,  # 负权重，越小越好
            'strain_energy': -0.5,  # 负权重
            'logp': 0.2,  # 适中的LogP更好
            'mqed': 1.0,  # 正权重
            'validity': 5.0,  # 有效性最重要
            'sims_surf': 0.8,  # 表面相似性
            'sims_esp': 0.8,  # 静电势相似性
        }
    
    def compute_rmsd(self, mol_generated, mol_reference):
        """计算RMSD"""
        try:
            from rdkit.Chem import AllChem
            # 对齐分子
            rmsd = AllChem.AlignMol(mol_generated, mol_reference)
            return rmsd
        except:
            return 10.0  # 失败时返回高RMSD
    
    
    def check_validity(self, mol):
        """检查分子有效性"""
        if mol is None:
            return 0.0
        try:
            # 检查基本化学有效性
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            if smiles:
                return 1.0
        except:
            pass
        return 0.0
    
    
    def score_molecule(self, mol_generated, mol_reference=None):
        """
        综合评分一个分子（使用完整的Shepherd Score）
        参考：evaluation/main.ipynb
        
        Returns:
            dict: 包含详细评分的字典，包括total_score
        """
        if mol_generated is None:
            return {'total_score': -999.0, 'is_valid': False}
        
        try:
            from shepherd_score.evaluations.evaluate import ConfEval, ConditionalEvalPipeline
            from shepherd_score.container import Molecule
            
            # 基本有效性检查
            scores = {}
            scores['is_valid'] = self.check_validity(mol_generated)
            if scores['is_valid'] < 0.5:
                return {'total_score': -999.0, 'is_valid': False}
            
            # 提取原子和坐标
            atoms = np.array([a.GetAtomicNum() for a in mol_generated.GetAtoms()])
            positions = mol_generated.GetConformer().GetPositions()
            
            # 使用ConfEval进行构象评估
            conf_eval = ConfEval(atoms, positions, solvent='water')
            eval_df = conf_eval.to_pandas()
            
            # 提取关键指标
            scores['strain_energy'] = float(eval_df['strain_energies'].iloc[0]) if 'strain_energies' in eval_df else 0.0
            scores['logp'] = float(eval_df['logPs'].iloc[0]) if 'logPs' in eval_df else 0.0
            scores['mqed'] = float(eval_df['QEDs'].iloc[0]) if 'QEDs' in eval_df else 0.0
            scores['sa_score'] = float(eval_df['SA_scores'].iloc[0]) if 'SA_scores' in eval_df else 5.0
            scores['fsp3'] = float(eval_df['fsp3s'].iloc[0]) if 'fsp3s' in eval_df else 0.0
            
            # 如果有参考分子，使用ConditionalEvalPipeline计算相似性
            if mol_reference is not None:
                try:
                    # 创建参考分子的Molecule对象
                    ref_molec = Molecule(
                        mol_reference, 
                        num_surf_points=200, 
                        probe_radius=1.2,
                        partial_charges=None, 
                        pharm_multi_vector=False
                    )
                    
                    # 创建生成分子列表
                    generated_mols_list = [(atoms, positions)]
                    
                    # 使用ConditionalEvalPipeline评估
                    cond_pipe = ConditionalEvalPipeline(
                        ref_molec,
                        generated_mols=generated_mols_list,
                        condition='all',
                        num_surf_points=200,
                        pharm_multi_vector=False,
                        solvent=None
                    )
                    cond_pipe.evaluate(verbose=False)
                    
                    # 获取评估结果
                    properties_df_cond, global_attr_cond = cond_pipe.to_pandas()
                    
                    # 提取相似性指标
                    if len(global_attr_cond) > 0:
                        scores['rmsd'] = float(global_attr_cond['rmsds'].iloc[0]) if 'rmsds' in global_attr_cond else 10.0
                        scores['sims_surf_upper_bound'] = float(properties_df_cond['sims_surf_upper_bound']) if 'sims_surf_upper_bound' in properties_df_cond else 0.0
                        scores['sims_esp_upper_bound'] = float(properties_df_cond['sims_esp_upper_bound']) if 'sims_esp_upper_bound' in properties_df_cond else 0.0
                except Exception as e:
                    print(f"相似性计算失败: {e}")
                    scores['rmsd'] = 10.0
                    scores['sims_surf_upper_bound'] = 0.0
                    scores['sims_esp_upper_bound'] = 0.0
            
            # 计算综合得分
            # 权重策略：QED (2.0) + logP评分 (1.5) - strain (0.5) - SA (0.3) + 相似性 (1.0)
            total_score = scores['mqed'] * 2.0
            
            # LogP惩罚（理想范围 0-3）
            logp_penalty = abs(scores['logp'] - 1.5) * 0.3
            total_score -= logp_penalty
            
            # 应变能惩罚
            total_score -= min(scores['strain_energy'], 10.0) * 0.5
            
            # SA Score惩罚（越低越好）
            total_score -= scores['sa_score'] * 0.3
            
            # 相似性奖励
            if 'sims_surf_upper_bound' in scores:
                total_score += scores['sims_surf_upper_bound'] * 1.0
            if 'sims_esp_upper_bound' in scores:
                total_score += scores['sims_esp_upper_bound'] * 1.0
            
            # RMSD惩罚
            if 'rmsd' in scores:
                total_score -= min(scores['rmsd'], 5.0) * 0.5
            
            scores['total_score'] = total_score
            return scores
            
        except Exception as e:
            print(f"分子评分失败: {e}")
            import traceback
            traceback.print_exc()
            return {'total_score': -999.0, 'is_valid': False, 'error': str(e)}
    
    def score_batch(self, mols_generated, mol_reference=None):
        """批量评分"""
        scores = []
        for mol in mols_generated:
            score = self.score_molecule(mol, mol_reference)
            scores.append(score)
        return scores


class OnlineSampler:
    """
    在线采样器：使用当前模型生成新分子
    """
    
    def __init__(self, model_pl, params, dataset=None, device='cuda'):
        """
        Args:
            model_pl: LightningModule对象（包含model和params）
            params: 参数字典
            dataset: 训练数据集（用于获取marginals）
            device: 设备（cuda或cpu）
        """
        self.model_pl = model_pl  # 保存LightningModule用于inference_sample
        self.model = model_pl.model  # 保存model用于eval/train切换
        self.params = params
        self.device = device
        self.dataset = dataset
        
        # 从dataset获取marginals（用于inference_sample）
        if dataset is not None:
            # 正确的属性名是 'marginals' 而不是 'x_marginals'
            # ⚠️ 关键：将marginals移到GPU，避免后续CPU/GPU设备冲突
            import torch
            self.atom_marginals = dataset.x1_atom_diffuser.transition_model.marginals.to(device) if hasattr(dataset, 'x1_atom_diffuser') else None
            self.bond_marginals = dataset.x1_bond_diffuser.transition_model.marginals.to(device) if hasattr(dataset, 'x1_bond_diffuser') else None
            self.pharm_marginals = dataset.x4_pharm_diffuser.transition_model.marginals.to(device) if hasattr(dataset, 'x4_pharm_diffuser') else None
        else:
            self.atom_marginals = None
            self.bond_marginals = None
            self.pharm_marginals = None
        
        # 采样参数
        self.num_samples_per_mol = 4  # 每个种子分子生成4个样本
        self.timesteps = params.get('sampling', {}).get('timesteps', 1000)
    
    def sample_from_seed(self, seed_mol_data):
        """
        从一个种子分子生成多个新分子
        
        Args:
            seed_mol_data: 种子分子的数据（HeteroData格式）
        
        Returns:
            List[rdkit.Mol]: 生成的分子列表
        """
        print(f"\n  📌 开始从种子分子生成 {self.num_samples_per_mol} 个样本...")
        generated_mols = []
        
        # 将模型设为评估模式
        self.model.eval()
        
        with torch.no_grad():
            for i in range(self.num_samples_per_mol):
                try:
                    print(f"    🔬 样本 {i+1}/{self.num_samples_per_mol}: 开始扩散采样...")
                    # 使用完整的inference_sample进行条件生成
                    mol = self._sample_one_molecule(seed_mol_data)
                    
                    if mol is not None:
                        print(f"    ✅ 样本 {i+1}/{self.num_samples_per_mol}: 成功生成分子")
                    else:
                        print(f"    ⚠️  样本 {i+1}/{self.num_samples_per_mol}: 生成失败")
                    
                    generated_mols.append(mol)
                    
                except Exception as e:
                    print(f"    ❌ 样本 {i+1}/{self.num_samples_per_mol}: 采样失败 - {e}")
                    import traceback
                    traceback.print_exc()
                    generated_mols.append(None)
        
        # 恢复训练模式
        self.model.train()
        
        valid_count = sum(1 for mol in generated_mols if mol is not None)
        print(f"  📊 种子分子采样完成: {valid_count}/{self.num_samples_per_mol} 个有效分子\n")
        
        return generated_mols
    
    def _sample_one_molecule(self, seed_data):
        """
        采样单个分子 - 使用完整的inference_sample逻辑
        参考：evaluation/main.ipynb
        
        Args:
            seed_data: 种子分子数据 (mol_block, charges)
        
        Returns:
            rdkit.Mol: 生成的分子对象（如果成功），否则返回None
        """
        try:
            # 输出每个阶段
            print("  🔮 开始采样一个分子")
            
            # 1. 从种子数据创建分子
            print("    🧬 从种子数据创建分子")
            mol_block, charges = seed_data
            seed_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
            
            if seed_mol is None:
                return None
            
            # 2. 预处理分子坐标（中心化）
            print("    🪧 预处理分子坐标（中心化）")
            mol_coordinates = np.array(seed_mol.GetConformer().GetPositions())
            mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
            seed_mol = update_mol_coordinates(seed_mol, mol_coordinates)
            
            # 3. 提取条件特征（参考evaluation/main.ipynb）
            print("    🔍 提取条件特征")
            centers = seed_mol.GetConformer().GetPositions()
            radii = get_atomic_vdw_radii(seed_mol)
            
            # 生成分子表面点云（x2模态）
            print("    📈 生成分子表面点云")
            surface = get_molecular_surface(
                centers, 
                radii, 
                self.params['dataset']['x2']['num_points'],
                probe_radius=self.params['dataset']['probe_radius'],
                num_samples_per_atom=20,
            )
            
            # 提取药效团特征（x4模态）
            print("    🧵 提取药效团特征")
            pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
                seed_mol,
                multi_vector=self.params['dataset']['x4']['multivectors'],
                check_access=self.params['dataset']['x4']['check_accessibility'],
            )
            
            # 计算表面静电势（x3模态）
            print("    🔋 计算表面静电势")
            electrostatics = get_electrostatics_given_point_charges(
                charges, centers, surface,
            )
            
            # 4. 使用inference_sample生成新分子
            print("    🔁 使用inference_sample生成新分子")
            # 修复：使用固定的原子数而不是种子分子的原子数
            # n_atoms = len(seed_mol.GetAtoms())  # 原代码：可能只有15-30个原子
            n_atoms = self.params.get('sampling', {}).get('fixed_n_atoms', 70)  # 从配置读取，默认70
            print(f"    📌 使用固定原子数: {n_atoms} (种子分子实际原子数: {len(seed_mol.GetAtoms())})")
            num_pharmacophores = len(pharm_types)
            
            # inference_sample期望numpy数组输入，内部会转换为GPU tensor
            # 调用inference_sample（参考evaluation/main.ipynb的参数）
            generated_samples = inference_sample(
                self.model_pl,
                batch_size=1,  # 一次生成一个
                N_x1=n_atoms,
                N_x4=num_pharmacophores,
                unconditional=False,  # 条件生成
                
                # 噪声控制
                prior_noise_scale=1.0,
                denoising_noise_scale=1.0,
                inject_noise_at_ts=[],
                inject_noise_scales=[],
                
                # 谐波化参数
                harmonize=False,
                harmonize_ts=[],
                harmonize_jumps=[],
                
                # 条件修复（inpainting）参数
                inpaint_x2_pos=False,
                inpaint_x3_pos=False,
                inpaint_x3_x=False,
                inpaint_x4_pos=True,
                inpaint_x4_direction=True,
                inpaint_x4_type=True,
                
                # 修复时间控制
                stop_inpainting_at_time_x2=0.0,
                add_noise_to_inpainted_x2_pos=0.0,
                stop_inpainting_at_time_x3=0.0,
                add_noise_to_inpainted_x3_pos=0.0,
                add_noise_to_inpainted_x3_x=0.0,
                stop_inpainting_at_time_x4=0.0,
                add_noise_to_inpainted_x4_pos=0.0,
                add_noise_to_inpainted_x4_direction=0.0,
                add_noise_to_inpainted_x4_type=0.0,
                
                # 条件输入（保持numpy数组格式）
                center_of_mass=np.zeros(3),
                surface=surface,
                electrostatics=electrostatics,
                pharm_types=pharm_types,
                pharm_pos=pharm_pos,
                pharm_direction=pharm_direction,
                
                # 边际分布（从dataset获取）
                atom_marginals=self.atom_marginals,
                bond_marginals=self.bond_marginals,
                pharm_marginals=self.pharm_marginals,
            )
            
            # 5. 提取生成的分子
            print(f"      🔄 采样完成，正在转换为RDKit分子...")
            if len(generated_samples) > 0:
                sample_dict = generated_samples[0]
                generated_mol = create_rdkit_molecule(sample_dict)

                print("创建函数结束")
                
                if generated_mol is not None:
                    try:
                        print(f"      ✓ RDKit分子创建成功，正在验证...")
                        Chem.SanitizeMol(generated_mol)
                        print(f"      ✓ 分子验证通过")
                        return generated_mol
                    except Exception as e:
                        print(f"      ✗ 分子验证失败: {e}")
                        return None
                else:
                    print(f"      ✗ RDKit分子创建失败")
            
            return None
                
        except Exception as e:
            print(f"采样过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def batch_sample(self, seed_mol_list, show_progress=True):
        """
        批量采样
        
        Args:
            seed_mol_list: 种子分子列表
            
        Returns:
            List[List[rdkit.Mol]]: 每个种子对应的生成分子列表
        """
        all_generated = []
        
        iterator = tqdm(seed_mol_list, desc="在线采样") if show_progress else seed_mol_list
        
        for seed_data in iterator:
            mols = self.sample_from_seed(seed_data)
            all_generated.append(mols)
        
        return all_generated


class PreferencePairBuilder:
    """
    偏好对构建器：从生成的分子中构建(winner, loser)偏好对
    """
    
    def __init__(self, scorer: ShepherdScorer, params):
        self.scorer = scorer
        self.params = params
        
        # 偏好对筛选阈值
        self.min_score_gap = params.get('dpo', {}).get('min_score_gap', 0.5)
    
    def build_pairs_from_samples(self, generated_mols, reference_mol=None):
        """
        从生成的样本中构建偏好对
        
        Args:
            generated_mols: List[rdkit.Mol]，生成的分子列表（通常是4个）
            reference_mol: 参考分子（可选）
        
        Returns:
            Optional[Tuple]: (winner_mol, loser_mol, score_dict_winner, score_dict_loser)
                             如果无法构建有效偏好对则返回None
        """
        # 过滤掉None
        valid_mols = [mol for mol in generated_mols if mol is not None]
        
        print(f"  🎯 开始评分: {len(valid_mols)} 个有效分子")
        if len(valid_mols) < 2:
            print(f"  ⚠️  分子数量不足，无法构建偏好对")
            return None
        
        # 评分（返回字典列表）
        print(f"  ⏳ 使用Shepherd Score评分系统评分中...")
        score_dicts = self.scorer.score_batch(valid_mols, reference_mol)
        
        # 过滤掉无效分子
        valid_pairs = [(mol, score_dict) for mol, score_dict in zip(valid_mols, score_dicts) 
                       if score_dict.get('is_valid', False)]
        
        print(f"  📋 评分完成: {len(valid_pairs)}/{len(valid_mols)} 个分子评分有效")
        if len(valid_pairs) < 2:
            print(f"  ⚠️  有效评分不足2个，无法构建偏好对")
            return None
        
        # 按total_score排序
        valid_pairs.sort(key=lambda x: x[1]['total_score'], reverse=True)
        
        winner_mol, winner_score_dict = valid_pairs[0]
        loser_mol, loser_score_dict = valid_pairs[-1]
        
        # 检查分数差距是否足够大
        score_gap = winner_score_dict['total_score'] - loser_score_dict['total_score']
        print(f"  📊 最高分: {winner_score_dict['total_score']:.3f}, 最低分: {loser_score_dict['total_score']:.3f}, 分差: {score_gap:.3f}")
        
        if score_gap < self.min_score_gap:
            print(f"  ⚠️  分数差距 ({score_gap:.3f}) 小于阈值 ({self.min_score_gap})，舍弃该偏好对")
            return None
        
        print(f"  ✅ 成功构建偏好对 (分差: {score_gap:.3f})")
        return (winner_mol, loser_mol, winner_score_dict, loser_score_dict)
    
    def batch_build_pairs(self, all_generated_mols, reference_mols=None, show_progress=True):
        """
        批量构建偏好对
        
        Args:
            all_generated_mols: List[List[rdkit.Mol]]
            reference_mols: List[rdkit.Mol] 或 None
        
        Returns:
            List[Tuple]: 偏好对列表
        """
        pairs = []
        
        n = len(all_generated_mols)
        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="构建偏好对")
        
        for i in iterator:
            generated = all_generated_mols[i]
            ref_mol = reference_mols[i] if reference_mols else None
            
            pair = self.build_pairs_from_samples(generated, ref_mol)
            if pair is not None:
                pairs.append(pair)
        
        return pairs
    
    def filter_hard_samples(self, pairs_with_losses, implicit_acc_threshold=0.6):
        """
        筛选学习难度高的样本
        
        Args:
            pairs_with_losses: List[Tuple]，包含(pair, loss, implicit_acc)
            implicit_acc_threshold: float，隐式准确率阈值
        
        Returns:
            List: 过滤后的偏好对
        """
        hard_pairs = []
        
        for pair, loss, implicit_acc in pairs_with_losses:
            # 保留模型还没学好的样本
            if implicit_acc < implicit_acc_threshold:
                hard_pairs.append(pair)
        
        return hard_pairs


class DPOSamplingScheduler:
    """
    DPO采样调度器：管理采样比例和策略
    """
    
    def __init__(self, params):
        self.params = params
        self.base_sampling_ratio = params.get('dpo', {}).get('sampling_ratio', 0.05)
        
        # 分桶策略
        self.size_buckets = {
            'small': (0, 15),    # 原子数 <= 15
            'medium': (16, 35),  # 16 <= 原子数 <= 35
            'large': (36, 999),  # 原子数 > 35
        }
        
        self.bucket_weights = {
            'small': 0.3,
            'medium': 0.4,
            'large': 0.3,
        }
    
    def select_seeds(self, dataset, epoch, losses=None):
        """
        根据策略选择种子分子
        
        实现策略：
        1. 如果提供了losses，优先选择loss较高的样本（困难样本）
        2. 否则，使用分层随机采样（按分子大小分桶）
        
        Args:
            dataset: 训练数据集
            epoch: 当前epoch
            losses: Optional[Dict]，每个样本的损失
        
        Returns:
            List[int]: 选中的样本索引
        """
        n_total = len(dataset)
        n_samples = int(n_total * self.base_sampling_ratio)
        
        # 策略1：如果有loss信息，选择困难样本
        if losses is not None and len(losses) > 0:
            # 按loss降序排序，选择前n_samples个
            sorted_indices = sorted(losses.keys(), key=lambda k: losses[k], reverse=True)
            selected_indices = sorted_indices[:n_samples]
            return selected_indices
        
        # 策略2：分层随机采样（按分子大小分桶）
        # 这样可以确保大中小分子都有代表
        try:
            # 尝试获取分子大小信息
            molecule_sizes = []
            for i in range(min(n_total, 1000)):  # 采样一部分评估
                try:
                    data = dataset[i]
                    if hasattr(data, 'x1') and hasattr(data['x1'], 'pos'):
                        size = len(data['x1'].pos)
                        molecule_sizes.append((i, size))
                except:
                    continue
            
            if len(molecule_sizes) > 100:
                # 按大小分桶
                small_indices = [i for i, s in molecule_sizes if self.size_buckets['small'][0] <= s <= self.size_buckets['small'][1]]
                medium_indices = [i for i, s in molecule_sizes if self.size_buckets['medium'][0] <= s <= self.size_buckets['medium'][1]]
                large_indices = [i for i, s in molecule_sizes if self.size_buckets['large'][0] <= s <= self.size_buckets['large'][1]]
                
                # 按权重采样
                n_small = int(n_samples * self.bucket_weights['small'])
                n_medium = int(n_samples * self.bucket_weights['medium'])
                n_large = n_samples - n_small - n_medium
                
                selected_indices = []
                if len(small_indices) > 0:
                    selected_indices.extend(np.random.choice(small_indices, size=min(n_small, len(small_indices)), replace=False))
                if len(medium_indices) > 0:
                    selected_indices.extend(np.random.choice(medium_indices, size=min(n_medium, len(medium_indices)), replace=False))
                if len(large_indices) > 0:
                    selected_indices.extend(np.random.choice(large_indices, size=min(n_large, len(large_indices)), replace=False))
                
                # 如果不够，随机补充
                if len(selected_indices) < n_samples:
                    remaining = n_samples - len(selected_indices)
                    all_indices = set(range(n_total)) - set(selected_indices)
                    selected_indices.extend(np.random.choice(list(all_indices), size=remaining, replace=False))
                
                return selected_indices[:n_samples]
        except Exception as e:
            print(f"分层采样失败，回退到随机采样: {e}")
        
        # 策略3：后备方案 - 完全随机采样
        selected_indices = np.random.choice(n_total, size=n_samples, replace=False)
        return selected_indices.tolist()
