"""
Utility functions for ShEPhERD inference scaling.

This module provides helper functions for working with ShEPhERD's inference output.
"""

import json
import logging
from copy import deepcopy
import signal
from contextlib import contextmanager
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit.Geometry import Point3D

def edge_list_to_adjacency_matrix(num_atoms, edge_index, edge_types):
    """
    Convert edge list format to adjacency matrix format.
    
    Parameters:
    - num_atoms: int, number of atoms
    - edge_index: array of shape (2, num_edges) or (num_edges, 2) with edge indices
    - edge_types: array of shape (num_edges,) with edge type indices
    
    Returns:
    - adjacency_matrix: array of shape (num_atoms, num_atoms) with bond type indices
    """
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    
    # Ensure edge_index has shape (2, num_edges)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    
    # Fill adjacency matrix (upper triangular)
    for idx in range(edge_index.shape[1]):
        i, j = edge_index[0, idx], edge_index[1, idx]
        # Ensure upper triangular
        if i < j:
            adjacency_matrix[i, j] = edge_types[idx]
        else:
            adjacency_matrix[j, i] = edge_types[idx]
    
    return adjacency_matrix


def build_3d_mol_from_arrays(atom_type_array, bond_adjacent_array, positions_3d, bond_types=None):
    """
    Build a 3D RDKit molecule from atom types, bond adjacency array, and 3D positions.
    
    Parameters:
    - atom_type_array: array of shape (N,) with atomic numbers (1=H, 6=C, 7=N, 8=O, etc.)
    - bond_adjacent_array: array of shape (N, N) with bond type indices (upper triangular)
    - positions_3d: array of shape (N, 3) with 3D coordinates
    - bond_types: list of bond type strings (default: [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
    
    Returns:
    - RDKit Mol object with 3D coordinates
    """
    
    if bond_types is None:
        bond_types = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    
    # Create editable molecule
    mol = Chem.EditableMol(Chem.Mol())
    
    # Keep track of original to new atom indices (excluding H atoms)
    atom_idx_mapping = {}
    new_atom_idx = 0
    
    # Add atoms (skip hydrogen atoms with atomic number 1)
    for i, atomic_number in enumerate(atom_type_array):
        if atomic_number == 0 or atomic_number == 1:
            continue  # Skip None (0) or hydrogen atoms (1)
            
        # Create atom using atomic number
        atom = Chem.Atom(int(atomic_number))
        mol.AddAtom(atom)
        atom_idx_mapping[i] = new_atom_idx
        new_atom_idx += 1
    
    # Add bonds
    n_atoms = len(atom_type_array)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Skip if either atom was excluded (hydrogen or invalid)
            if i not in atom_idx_mapping or j not in atom_idx_mapping:
                continue
                
            bond_type_idx = bond_adjacent_array[i][j]
            if bond_type_idx == 0 or bond_type_idx >= len(bond_types):
                continue  # Skip None bonds or invalid indices
                
            bond_type_str = bond_types[bond_type_idx]
            
            # Convert bond type string to RDKit bond type
            if bond_type_str == 'SINGLE':
                bond_type = Chem.BondType.SINGLE
            elif bond_type_str == 'DOUBLE':
                bond_type = Chem.BondType.DOUBLE
            elif bond_type_str == 'TRIPLE':
                bond_type = Chem.BondType.TRIPLE
            elif bond_type_str == 'AROMATIC':
                bond_type = Chem.BondType.AROMATIC
            else:
                continue  # Skip unknown bond types
            
            mol.AddBond(atom_idx_mapping[i], atom_idx_mapping[j], bond_type)
    
    # Convert to molecule
    mol = mol.GetMol()
    
    if mol is None:
        return None
    
    # Add 3D coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for orig_idx, new_idx in atom_idx_mapping.items():
        x, y, z = positions_3d[orig_idx]
        conf.SetAtomPosition(new_idx, Point3D(float(x), float(y), float(z)))
    
    mol.AddConformer(conf)
    
    # Sanitize molecule
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"Warning: Could not sanitize molecule: {e}")
        print("Returning unsanitized molecule.")
    
    return mol



class TimeoutException(Exception):
    """超时异常"""
    pass

@contextmanager
def time_limit(seconds):
    """
    超时上下文管理器
    Args:
        seconds: 超时秒数
    """
    def signal_handler(signum, frame):
        raise TimeoutException("操作超时")
    
    # 设置信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # 取消alarm
        signal.alarm(0)


def get_xyz_content(sample):
    """
    Generate XYZ file content string from a ShEPhERD sample.
    
    Args:
        sample (dict): ShEPhERD output dictionary with x1 containing atoms and positions.
        
    Returns:
        str: XYZ file content or None if input is invalid.
    """
    if 'x1' not in sample or 'atoms' not in sample['x1'] or 'positions' not in sample['x1']:
        logging.warning("Invalid sample format for XYZ generation")
        return None

    try:
        atoms = sample['x1']['atoms']
        positions = sample['x1']['positions']

        xyz_lines = [f"{len(atoms)}", "Generated by ShEPhERD Inference Scaling"]
        for i in range(len(atoms)):
            try:
                atomic_number = int(atoms[i])
                # attempt to get symbol, default to element number if fails
                try:
                    symbol = Chem.Atom(atomic_number).GetSymbol()
                except Exception:
                    symbol = str(atomic_number)
                pos = positions[i]
                xyz_lines.append(f"{symbol} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}")
            except (ValueError, IndexError) as e:
                logging.warning(f"Skipping atom {i} due to data issue: {e}")
                continue

        xyz_lines[0] = str(len(xyz_lines) - 2)

        return "\n".join(xyz_lines)

    except Exception as e:
        logging.error(f"Error generating XYZ content: {e}")
        return None


def create_rdkit_molecule(sample):
    """
    Create an RDKit molecule from ShEPhERD output using XYZ block approach.
    
    Args:
        sample (dict): ShEPhERD output dictionary with x1 containing atoms and positions.
        
    Returns:
        rdkit.Chem.rdchem.Mol: RDKit molecule object or None if conversion fails.
    """
    print("=" * 60)
    print("🧪 开始创建RDKit分子对象")
    print("=" * 60)
    
    # 阶段1: 检查输入数据
    print("📋 阶段1/7: 检查输入数据...")
    if 'x1' not in sample:
        print("❌ 未找到原子数据(x1)")
        return None
    print("✓ 输入数据检查通过")

    # try:
    # 阶段2: 提取原子、位置和键数据
    print("\n📦 阶段2/7: 提取原子、位置和键数据...")
    atoms = sample['x1']['atoms']
    positions = sample['x1']['positions']
    bonds = sample['x1'].get('bonds', None)  # 键类型数据（edge list格式）
    print(f"✓ 提取到 {len(atoms)} 个原子, {len(positions)} 个坐标")
    if bonds is not None:
        print(f"✓ 提取到 {len(bonds)} 条键")
    else:
        print("⚠️  未找到键数据，将尝试使用备用方法")
    
    # 阶段3: 检查数据完整性
    print("\n🔍 阶段3/7: 检查数据完整性...")
    if len(atoms) == 0 or len(positions) == 0:
        print("❌ 原子或坐标数据为空")
        return None
        
    if len(atoms) != len(positions):
        print(f"❌ 原子数({len(atoms)})与坐标数({len(positions)})不匹配")
        return None
    print(f"✓ 数据完整性检查通过: {len(atoms)} 个原子")

    # 阶段4: 先构建原始的邻接矩阵（基于所有原子）
    print("\n📝 阶段4/7: 构建原始键邻接矩阵...")
    original_bond_adjacent_array = None
    
    if bonds is not None:
        try:
            # 根据inference.py的逻辑，bonds是基于所有原子的完全图上三角矩阵
            num_atoms_original = len(atoms)
            expected_edges = num_atoms_original * (num_atoms_original - 1) // 2
            
            print(f"原始原子数: {num_atoms_original}, 期望边数: {expected_edges}, 实际bonds数: {len(bonds)}")
            
            if len(bonds) == expected_edges:
                # 创建上三角邻接矩阵的edge_index
                edge_sources = []
                edge_targets = []
                for i in range(num_atoms_original):
                    for j in range(i + 1, num_atoms_original):
                        edge_sources.append(i)
                        edge_targets.append(j)
                
                edge_index = np.array([edge_sources, edge_targets])
                original_bond_adjacent_array = edge_list_to_adjacency_matrix(num_atoms_original, edge_index, bonds)
                print(f"✓ 成功构建原始键邻接矩阵（{np.sum(original_bond_adjacent_array > 0)}条键）")
            else:
                print(f"⚠️ 键数据长度({len(bonds)})与期望边数({expected_edges})不匹配")
        except Exception as e:
            print(f"⚠️ 构建原始邻接矩阵失败: {e}")
    
    # 阶段5: 过滤有效的原子、位置和对应的键
    print("\n🧹 阶段5/7: 过滤和验证原子数据...")
    valid_atoms = []
    valid_positions = []
    valid_indices = []  # 记录有效原子在原始数组中的索引
    invalid_count = 0
    
    for a in range(len(atoms)):
        try:
            atomic_number = int(atoms[a])
            position = positions[a]
            
            # 检查原子序数是否有效
            if atomic_number <= 0 or atomic_number > 118:
                print(f"Invalid atomic number: {atomic_number}")
                invalid_count += 1
                continue
            
            # 检查位置是否包含NaN或无穷值
            if np.any(np.isnan(position)) or np.any(np.isinf(position)):
                print(f"Atom {a} has invalid position (NaN/Inf): {position}")
                invalid_count += 1
                continue
            
            # 检查位置坐标是否合理（不能过大）
            if np.any(np.abs(position) > 1000):
                print(f"Atom {a} has unreasonable position: {position}")
                invalid_count += 1
                continue
            
            valid_atoms.append(atomic_number)
            valid_positions.append(position)
            valid_indices.append(a)
            
        except (ValueError, TypeError, IndexError) as e:
            print(f"Error processing atom {a}: {e}")
            invalid_count += 1
            continue
    
    # 统计并打印无效原子信息
    total_atoms = len(atoms)
    valid_count = len(valid_atoms)
    invalid_ratio = (invalid_count / total_atoms * 100) if total_atoms > 0 else 0
    
    if invalid_count > 0:
        print(f"📊 原子统计: 总数={total_atoms}, 有效={valid_count}, 无效={invalid_count} ({invalid_ratio:.1f}%)")
    else:
        print(f"✓ 所有 {valid_count} 个原子均有效")
    
    if len(valid_atoms) == 0:
        print("❌ 过滤后没有有效原子")
        return None
    
    # 从原始邻接矩阵中提取有效原子之间的键
    bond_adjacent_array = None
    if original_bond_adjacent_array is not None:
        try:
            print("提取有效原子之间的键...")
            num_valid = len(valid_atoms)
            bond_adjacent_array = np.zeros((num_valid, num_valid), dtype=int)
            
            for i in range(num_valid):
                for j in range(i + 1, num_valid):
                    orig_i = valid_indices[i]
                    orig_j = valid_indices[j]
                    bond_adjacent_array[i, j] = original_bond_adjacent_array[orig_i, orig_j]
            
            num_bonds = np.sum(bond_adjacent_array > 0)
            print(f"✓ 成功提取键邻接矩阵（{num_bonds}条键）")
        except Exception as e:
            print(f"⚠️ 提取键信息失败: {e}")
            bond_adjacent_array = None
    
    # 阶段6: 使用build_3d_mol_from_arrays创建分子
    print("\n🔬 阶段6/7: 使用build_3d_mol_from_arrays创建分子对象...")
    
    if bond_adjacent_array is None:
        print("❌ 键邻接矩阵不可用，跳过此分子")
        return None
    
    # 使用预测的键数据
    print("使用模型预测的键数据构建分子...")
    mol_final = build_3d_mol_from_arrays(
        atom_type_array=np.array(valid_atoms),
        bond_adjacent_array=bond_adjacent_array,
        positions_3d=np.array(valid_positions)
    )
    if mol_final is None:
        print("❌ 使用预测键数据创建分子失败，跳过此分子")
        return None
    print(f"✓ 分子对象创建成功（{mol_final.GetNumAtoms()}个原子，{mol_final.GetNumBonds()}条键）")
    
    # 阶段7: 验证分子
    print("\n🔗 阶段7/7: 验证分子...")
    
    print("\n✅ 化学键确定成功，开始验证分子...")
    # validate molecule
    try:
        radical_electrons = sum([a.GetNumRadicalElectrons() for a in mol_final.GetAtoms()])
        if radical_electrons > 0:
            print(f"⚠️  分子包含 {radical_electrons} 个自由基电子")
        
        mol_final.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol_final)
        print("✓ 分子验证成功")
    except Exception as e:
        print(f"❌ 分子验证失败: {e}")
        return None

    # try to generate SMILES to verify molecule
    print("\n🧬 生成SMILES并验证...")
    try:
        smiles = Chem.MolToSmiles(mol_final)
        print(f"✓ SMILES: {smiles}")
    except Exception as e:
        print(f"❌ SMILES生成失败: {e}")

    if '.' in smiles:
        print("❌ 分子是片段（包含'.'），创建失败")
        return None

    print("=" * 60)
    print("🎉 分子创建成功！")
    print("=" * 60)
    return mol_final
        
def create_rdkit_molecule_from_mol(atoms, positions, bonds=None):
    """
    Create an RDKit molecule from atoms, positions, and optional bonds data.
    
    Args:
        atoms: array of atomic numbers
        positions: array of 3D coordinates
        bonds: optional array of bond types (edge list format)
        
    Returns:
        rdkit.Chem.rdchem.Mol: RDKit molecule object or None if conversion fails.
    """
    # if 'x1' not in sample:
    #     logging.warning("No atom data (x1) found in sample")
    #     return None

    # try:
    # extract atoms and their positions from x1
    # atoms = sample['x1']['atoms']
    # positions = sample['x1']['positions']
    
    # 检查数据完整性
    if len(atoms) == 0 or len(positions) == 0:
        logging.warning("Empty atoms or positions data")
        return None
        
    if len(atoms) != len(positions):
        logging.warning(f"Mismatch between atoms ({len(atoms)}) and positions ({len(positions)}) count")
        return None

    # 先构建原始的邻接矩阵（基于所有原子）
    original_bond_adjacent_array = None
    
    if bonds is not None:
        try:
            num_atoms_original = len(atoms)
            expected_edges = num_atoms_original * (num_atoms_original - 1) // 2
            
            logging.info(f"原始原子数: {num_atoms_original}, 期望边数: {expected_edges}, 实际bonds数: {len(bonds)}")
            
            if len(bonds) == expected_edges:
                # 创建上三角邻接矩阵的edge_index
                edge_sources = []
                edge_targets = []
                for i in range(num_atoms_original):
                    for j in range(i + 1, num_atoms_original):
                        edge_sources.append(i)
                        edge_targets.append(j)
                
                edge_index = np.array([edge_sources, edge_targets])
                original_bond_adjacent_array = edge_list_to_adjacency_matrix(num_atoms_original, edge_index, bonds)
                logging.info(f"成功构建原始键邻接矩阵（{np.sum(original_bond_adjacent_array > 0)}条键）")
            else:
                logging.warning(f"键数据长度({len(bonds)})与期望边数({expected_edges})不匹配")
        except Exception as e:
            logging.warning(f"构建原始邻接矩阵失败: {e}")

    # 过滤有效的原子、位置和对应的键
    valid_atoms = []
    valid_positions = []
    valid_indices = []  # 记录有效原子在原始数组中的索引
    invalid_count = 0
    
    for a in range(len(atoms)):
        try:
            atomic_number = int(atoms[a])
            position = positions[a]
            
            # 检查原子序数是否有效
            if atomic_number <= 0 or atomic_number > 118:
                print(f"Invalid atomic number: {atomic_number}")
                invalid_count += 1
                continue
            
            # 检查位置是否包含NaN或无穷值
            if np.any(np.isnan(position)) or np.any(np.isinf(position)):
                print(f"Atom {a} has invalid position (NaN/Inf): {position}")
                invalid_count += 1
                continue
            
            # 检查位置坐标是否合理（不能过大）
            if np.any(np.abs(position) > 1000):
                print(f"Atom {a} has unreasonable position: {position}")
                invalid_count += 1
                continue
            
            valid_atoms.append(atomic_number)
            valid_positions.append(position)
            valid_indices.append(a)
            
        except (ValueError, TypeError, IndexError) as e:
            print(f"Error processing atom {a}: {e}")
            invalid_count += 1
            continue
    
    # 统计并打印无效原子信息
    total_atoms = len(atoms)
    valid_count = len(valid_atoms)
    invalid_ratio = (invalid_count / total_atoms * 100) if total_atoms > 0 else 0
    
    if invalid_count > 0:
        logging.info(f"📊 原子统计: 总数={total_atoms}, 有效={valid_count}, 无效={invalid_count} ({invalid_ratio:.1f}%)")
    
    if len(valid_atoms) == 0:
        logging.warning("No valid atoms found after filtering")
        return None

    # 从原始邻接矩阵中提取有效原子之间的键
    bond_adjacent_array = None
    if original_bond_adjacent_array is not None:
        try:
            logging.info("提取有效原子之间的键...")
            num_valid = len(valid_atoms)
            bond_adjacent_array = np.zeros((num_valid, num_valid), dtype=int)
            
            for i in range(num_valid):
                for j in range(i + 1, num_valid):
                    orig_i = valid_indices[i]
                    orig_j = valid_indices[j]
                    bond_adjacent_array[i, j] = original_bond_adjacent_array[orig_i, orig_j]
            
            num_bonds = np.sum(bond_adjacent_array > 0)
            logging.info(f"成功提取键邻接矩阵（{num_bonds}条键）")
        except Exception as e:
            logging.warning(f"提取键信息失败: {e}")
            bond_adjacent_array = None
    
    # 使用build_3d_mol_from_arrays创建分子
    if bond_adjacent_array is None:
        logging.warning("键邻接矩阵不可用，跳过此分子")
        return None
    
    # 使用预测的键数据
    logging.info("使用模型预测的键数据构建分子...")
    mol_final = build_3d_mol_from_arrays(
        atom_type_array=np.array(valid_atoms),
        bond_adjacent_array=bond_adjacent_array,
        positions_3d=np.array(valid_positions)
    )
    if mol_final is None:
        logging.warning("使用预测键数据创建分子失败，跳过此分子")
        return None
    logging.info(f"分子对象创建成功（{mol_final.GetNumAtoms()}个原子，{mol_final.GetNumBonds()}条键）")
    
    # validate molecule
    try:
        radical_electrons = sum([a.GetNumRadicalElectrons() for a in mol_final.GetAtoms()])
        if radical_electrons > 0:
            logging.warning(f"Molecule has {radical_electrons} radical electrons")
        
        mol_final.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol_final)
        logging.debug("Molecule validation successful")
    except Exception as e:
        logging.warning(f"Molecule validation failed: {e}")
        return None

    # try to generate SMILES to verify molecule
    try:
        smiles = Chem.MolToSmiles(mol_final)
        logging.debug(f"Generated SMILES: {smiles}")
    except Exception as e:
        logging.warning(f"SMILES generation failed: {e}")

    if '.' in smiles:
        logging.warning("Molecule is a fragment, failed to create molecule")
        return None

    return mol_final
        
    # except Exception as e:
    #     logging.warning(f"Error creating molecule: {e}")
    #     return None
