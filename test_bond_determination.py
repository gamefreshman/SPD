#!/usr/bin/env python3
"""
测试脚本：验证键确定方法的修改

测试 convert_data.py 中的新键确定逻辑
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/score'))

from shepherd_score.evaluations.utils.convert_data import get_mol_from_atom_pos

def test_without_bonds():
    """测试不提供键信息的情况（向后兼容）"""
    print("=" * 60)
    print("测试 1: 不提供键信息（使用 rdDetermineBonds）")
    print("=" * 60)
    
    # 简单的甲烷分子: CH4
    atoms = np.array([6, 1, 1, 1, 1])  # C, H, H, H, H
    positions = np.array([
        [0.0, 0.0, 0.0],      # C
        [0.63, 0.63, 0.63],   # H
        [-0.63, -0.63, 0.63], # H
        [-0.63, 0.63, -0.63], # H
        [0.63, -0.63, -0.63]  # H
    ])
    
    try:
        mol, charge, xyz_block = get_mol_from_atom_pos(atoms=atoms, positions=positions)
        if mol is not None:
            print(f"✓ 成功创建分子")
            print(f"  原子数: {mol.GetNumAtoms()}")
            print(f"  键数: {mol.GetNumBonds()}")
            print(f"  电荷: {charge}")
        else:
            print("✗ 分子创建失败")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
    
    print()

def test_with_bonds():
    """测试提供键信息的情况"""
    print("=" * 60)
    print("测试 2: 提供键信息（使用 build_3d_mol_from_arrays）")
    print("=" * 60)
    
    # 简单的乙烷分子: C2H6 (只包含重原子)
    atoms = np.array([6, 6])  # C, C
    positions = np.array([
        [0.0, 0.0, 0.0],   # C1
        [1.54, 0.0, 0.0]   # C2 (C-C键长约1.54Å)
    ])
    
    # 键信息：上三角矩阵格式
    # 对于2个原子，edge list长度为 2*(2-1)/2 = 1
    # bonds[0] 表示原子0和原子1之间的键：1表示单键
    bonds = np.array([1])  # C-C单键
    
    try:
        mol, charge, xyz_block = get_mol_from_atom_pos(
            atoms=atoms, 
            positions=positions,
            bonds=bonds
        )
        if mol is not None:
            print(f"✓ 成功创建分子")
            print(f"  原子数: {mol.GetNumAtoms()}")
            print(f"  键数: {mol.GetNumBonds()}")
            print(f"  电荷: {charge}")
            
            # 检查键类型
            if mol.GetNumBonds() > 0:
                bond = mol.GetBondWithIdx(0)
                print(f"  键类型: {bond.GetBondType()}")
        else:
            print("✗ 分子创建失败")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
    
    print()

def test_with_invalid_bonds():
    """测试提供无效键信息的情况（应该回退到 rdDetermineBonds）"""
    print("=" * 60)
    print("测试 3: 提供无效键信息（应回退到 rdDetermineBonds）")
    print("=" * 60)
    
    atoms = np.array([6, 1, 1, 1, 1])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63]
    ])
    
    # 提供错误长度的键数据
    bonds = np.array([1, 2, 3])  # 错误长度，应该是 5*4/2=10
    
    try:
        mol, charge, xyz_block = get_mol_from_atom_pos(
            atoms=atoms,
            positions=positions,
            bonds=bonds
        )
        if mol is not None:
            print(f"✓ 回退成功，创建了分子")
            print(f"  原子数: {mol.GetNumAtoms()}")
            print(f"  键数: {mol.GetNumBonds()}")
        else:
            print("✗ 分子创建失败")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("键确定方法测试")
    print("=" * 60 + "\n")
    
    test_without_bonds()
    test_with_bonds()
    test_with_invalid_bonds()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
