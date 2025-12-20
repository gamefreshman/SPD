#!/usr/bin/env python3
"""
DPO评估模块
从dpo_trainer.py提取的评估功能，用于单独评估生成的分子
"""

import os
import sys
import json
import argparse
import warnings
from typing import List, Dict, Tuple, Any
from collections import defaultdict

import numpy as np
import rdkit
import rdkit.Chem
from rdkit.Chem import Descriptors, QED

# Shepherd Score评估模块
try:
    from shepherd_score.evaluations.evaluate import ConfEval, ConditionalEvalPipeline
    from shepherd_score.container import Molecule
    SHEPHERD_SCORE_AVAILABLE = True
except ImportError:
    SHEPHERD_SCORE_AVAILABLE = False
    warnings.warn("shepherd_score模块未安装，将使用RDKit备选评估方案")

# 项目模块
try:
    from shepherd.extract_shepherd import create_rdkit_molecule
    from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
    from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
    from shepherd.shepherd_score_utils.generate_point_cloud import (
        get_atomic_vdw_radii,
        get_molecular_surface,
        get_electrostatics_given_point_charges,
    )
    SHEPHERD_UTILS_AVAILABLE = True
except ImportError:
    SHEPHERD_UTILS_AVAILABLE = False
    warnings.warn("shepherd工具模块未完全导入，部分功能可能受限")

# 警告过滤
warnings.filterwarnings("ignore", category=UserWarning)


class MoleculeJudge:
    """分子评估器"""
    
    def __init__(self, use_shepherd_score=True, verbose=True):
        """
        初始化评估器
        
        Args:
            use_shepherd_score: 是否使用Shepherd Score（需要xtb）
            verbose: 是否输出详细信息
        """
        self.use_shepherd_score = use_shepherd_score and SHEPHERD_SCORE_AVAILABLE
        self.verbose = verbose
        
        if self.use_shepherd_score:
            print("✅ 使用Shepherd Score评估（包含xtb计算）")
        else:
            print("⚠️  使用RDKit备选评估方案（不含应变能计算）")
    
    def evaluate_single_molecule(self, sample: Dict, sample_idx: int) -> Dict:
        """
        评估单个分子样本
        
        Args:
            sample: 分子样本数据（包含x1等字段）
            sample_idx: 样本索引
            
        Returns:
            评估结果字典，包含conf_scores和status
        """
        try:
            # 提取原子和坐标
            atoms = sample['x1']['atoms']
            positions = sample['x1']['positions']
            
            if isinstance(atoms, list):
                atoms = np.array(atoms).flatten()
            elif isinstance(atoms, np.ndarray):
                atoms = atoms.flatten()
            
            positions = np.array(positions)
            
            if len(atoms) == 0:
                return {'status': 'error', 'message': '原子数为0'}
            
            if self.verbose:
                print(f"  🔬 样本 {sample_idx}: {len(atoms)} 个原子")
            
            # 构建RDKit分子
            rdkit_mol = self._create_rdkit_mol_from_sample(atoms, positions)
            
            if rdkit_mol is None:
                return {'status': 'error', 'message': 'RDKit分子创建失败'}
            
            # 评估分子
            conf_scores = self._compute_conf_scores(rdkit_mol, atoms, positions)
            
            if conf_scores is None:
                return {'status': 'error', 'message': '评估失败'}
            
            # 计算SMILES
            try:
                smiles = rdkit.Chem.MolToSmiles(rdkit_mol)
                conf_scores['smiles'] = smiles
            except Exception:
                conf_scores['smiles'] = 'N/A'
            
            if self.verbose:
                print(f"     ✓ 评估完成: QED={conf_scores['qed']:.3f}, "
                      f"LogP={conf_scores['logp']:.2f}, "
                      f"SA={conf_scores['sa_score']:.2f}")
            
            return {
                'status': 'success',
                'conf_scores': conf_scores,
                'rdkit_mol': rdkit_mol,
                'atoms': atoms,
                'positions': positions,
            }
            
        except Exception as e:
            if self.verbose:
                print(f"     ✗ 评估失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _create_rdkit_mol_from_sample(self, atoms: np.ndarray, positions: np.ndarray):
        """从原子和坐标创建RDKit分子"""
        try:
            # 如果有create_rdkit_molecule函数，使用它
            if SHEPHERD_UTILS_AVAILABLE:
                sample_dict = {
                    'x1': {
                        'atoms': atoms,
                        'positions': positions,
                    }
                }
                return create_rdkit_molecule(sample_dict)
            else:
                # 简单的备选方案：创建基本分子结构
                mol = rdkit.Chem.RWMol()
                
                # 原子类型映射（根据SPD项目的atom_types）
                atom_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
                           15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
                
                # 添加原子
                for atom_num in atoms:
                    atom_symbol = atom_map.get(int(atom_num), 'C')
                    atom = rdkit.Chem.Atom(atom_symbol)
                    mol.AddAtom(atom)
                
                # 添加坐标
                conf = rdkit.Chem.Conformer(len(atoms))
                for i, pos in enumerate(positions):
                    conf.SetAtomPosition(i, tuple(pos))
                mol.AddConformer(conf)
                
                # 尝试推断键
                rdkit.Chem.SanitizeMol(mol, catchErrors=True)
                
                return mol.GetMol()
                
        except Exception as e:
            if self.verbose:
                print(f"       警告: RDKit分子创建失败 - {e}")
            return None
    
    def _compute_conf_scores(self, rdkit_mol, atoms, positions) -> Dict:
        """计算分子的构象评分"""
        
        # 优先尝试使用Shepherd Score的ConfEval
        if self.use_shepherd_score:
            try:
                conf_eval = ConfEval(atoms, positions, solvent='water')
                eval_df = conf_eval.to_pandas()
                print("       完整评估结果:", eval_df)
                
                conf_scores = {
                    'qed': float(eval_df['QEDs'].iloc[0]) if 'QEDs' in eval_df else 0.0,
                    'logp': float(eval_df['logPs'].iloc[0]) if 'logPs' in eval_df else 0.0,
                    'strain_energy': float(eval_df['strain_energies'].iloc[0]) if 'strain_energies' in eval_df else 0.0,
                    'sa_score': float(eval_df['SA_scores'].iloc[0]) if 'SA_scores' in eval_df else 5.0,
                }
                
                if self.verbose:
                    print(f"       使用ConfEval（含应变能）")
                
                return conf_scores
                
            except Exception as e:
                if self.verbose:
                    print(f"       ConfEval失败: {str(e)[:50]}..., 切换到RDKit")
        
        # 备选方案：使用RDKit
        try:
            qed_value = QED.qed(rdkit_mol)
            logp_value = Descriptors.MolLogP(rdkit_mol)
            
            # SA Score计算
            try:
                from rdkit.Chem import RDConfig
                sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
                import sascorer
                sa_value = sascorer.calculateScore(rdkit_mol)
            except Exception:
                sa_value = 5.0
            
            conf_scores = {
                'qed': qed_value,
                'logp': logp_value,
                'strain_energy': 0.0,  # RDKit无法计算，设为0
                'sa_score': sa_value,
            }
            
            if self.verbose:
                print(f"       使用RDKit（无应变能）")
            
            return conf_scores
            
        except Exception as e:
            if self.verbose:
                print(f"       RDKit评估也失败: {e}")
            return None
    
    def evaluate_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        批量评估分子
        
        Args:
            samples: 分子样本列表
            
        Returns:
            评估结果列表
        """
        print(f"\n🔍 开始评估 {len(samples)} 个分子样本...")
        
        evaluated_results = []
        success_count = 0
        
        for i, sample in enumerate(samples):
            result = self.evaluate_single_molecule(sample, i + 1)
            
            if result['status'] == 'success':
                success_count += 1
                evaluated_results.append(result)
        
        print(f"\n✅ 评估完成: {success_count}/{len(samples)} 个样本成功")
        
        return evaluated_results
    
    def compute_total_score(self, conf_scores: Dict, cond_scores: Dict = None) -> float:
        """
        计算综合分数（Shepherd Score风格）
        
        Args:
            conf_scores: 构象评分
            cond_scores: 条件评分（可选）
            
        Returns:
            总分
        """
        # 基础分数（构象质量）
        total_score = conf_scores['qed'] * 2.0
        total_score -= abs(conf_scores['logp'] - 1.5) * 0.3
        total_score -= min(conf_scores['strain_energy'], 10.0) * 0.5
        total_score -= conf_scores['sa_score'] * 0.3
        
        # 条件分数（如果提供）
        if cond_scores is not None:
            total_score += cond_scores.get('sims_surf', 0.0) * 1.0
            total_score += cond_scores.get('sims_esp', 0.0) * 1.0
            total_score -= min(cond_scores.get('rmsd', 5.0), 5.0) * 0.5
        
        return total_score
    
    def rank_molecules(self, evaluated_results: List[Dict]) -> List[Tuple[int, Dict, float]]:
        """
        对分子进行排名
        
        Args:
            evaluated_results: 评估结果列表
            
        Returns:
            排名列表 [(rank, result, total_score), ...]
        """
        # 计算总分
        scored_results = []
        for result in evaluated_results:
            if result['status'] == 'success':
                total_score = self.compute_total_score(result['conf_scores'])
                scored_results.append((result, total_score))
        
        # 排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 添加排名
        ranked_results = []
        for rank, (result, score) in enumerate(scored_results, 1):
            ranked_results.append((rank, result, score))
        
        return ranked_results
    
    def print_ranking_report(self, ranked_results: List[Tuple[int, Dict, float]]):
        """打印排名报告"""
        print("\n" + "="*80)
        print("📊 分子质量排名")
        print("="*80)
        
        print(f"\n{'排名':<6} {'总分':<8} {'QED':<8} {'LogP':<8} {'应变能':<10} {'SA分数':<8} {'SMILES'}")
        print("-" * 80)
        
        for rank, result, total_score in ranked_results[:20]:  # 只显示前20个
            conf = result['conf_scores']
            smiles = conf.get('smiles', 'N/A')
            smiles_short = smiles[:40] + '...' if len(smiles) > 40 else smiles
            
            print(f"{rank:<6} {total_score:<8.3f} {conf['qed']:<8.3f} {conf['logp']:<8.2f} "
                  f"{conf['strain_energy']:<10.3f} {conf['sa_score']:<8.2f} {smiles_short}")
        
        if len(ranked_results) > 20:
            print(f"\n... 还有 {len(ranked_results) - 20} 个分子")
    
    def save_results(self, ranked_results: List[Tuple[int, Dict, float]], output_file: str):
        """保存评估结果到JSON文件"""
        results_for_json = []
        
        for rank, result, total_score in ranked_results:
            conf = result['conf_scores']
            
            item = {
                'rank': rank,
                'total_score': float(total_score),
                'qed': float(conf['qed']),
                'logp': float(conf['logp']),
                'strain_energy': float(conf['strain_energy']),
                'sa_score': float(conf['sa_score']),
                'smiles': conf.get('smiles', 'N/A'),
                'num_atoms': int(len(result['atoms'])),
            }
            
            results_for_json.append(item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 评估结果已保存到: {output_file}")


def load_molecules_from_json(json_file: str) -> List[Dict]:
    """从JSON文件加载分子"""
    print(f"\n📂 加载分子文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        molecules = json.load(f)
    
    print(f"✅ 成功加载 {len(molecules)} 个分子")
    
    return molecules


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DPO分子评估工具")
    parser.add_argument('input_file', type=str, help='输入的JSON文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='输出结果文件路径（默认：input_file_evaluated.json）')
    parser.add_argument('--use-shepherd', action='store_true', default=False,
                       help='使用Shepherd Score（需要xtb）')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='显示详细信息')
    parser.add_argument('--top-k', type=int, default=20,
                       help='显示前K个最佳分子（默认：20）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"❌ 文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 确定输出文件
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_evaluated.json"
    else:
        output_file = args.output
    
    print("="*80)
    print("🧬 DPO分子评估系统")
    print("="*80)
    
    # 加载分子
    molecules = load_molecules_from_json(args.input_file)
    
    # 创建评估器
    judge = MoleculeJudge(use_shepherd_score=args.use_shepherd, verbose=args.verbose)
    
    # 评估分子
    evaluated_results = judge.evaluate_batch(molecules)
    
    if len(evaluated_results) == 0:
        print("❌ 没有成功评估的分子")
        sys.exit(1)
    
    # 排名
    ranked_results = judge.rank_molecules(evaluated_results)
    
    # 打印报告
    judge.print_ranking_report(ranked_results)
    
    # 保存结果
    judge.save_results(ranked_results, output_file)
    
    print("\n" + "="*80)
    print("✅ 评估完成")
    print("="*80)


if __name__ == '__main__':
    main()
