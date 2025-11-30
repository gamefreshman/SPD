"""
Helper functions to convert different data types.
"""

import os
import sys
from typing import Union, Tuple, List, Optional
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
import rdkit.Chem.rdDetermineBonds
import pandas as pd

# 导入 extract.py 中的函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../shepherd'))
from extract import build_3d_mol_from_arrays, edge_list_to_adjacency_matrix


def write_xyz_file(atomic_numbers: np.ndarray,
                   positions: np.ndarray,
                   path_to_file: Union[str, None] = None
                   ) -> str:
    """
    Writes an xyz file of an atomistic structure, given np.ndarray of atomic numbers and coordinates.
    
    Arguments
    ---------
    atomic_numbers : np.ndarray of shape (N,) containing atomic numbers
    positions : np.ndarray of shape (N,3) containing atomic coordinates
    path_to_file : str specifying file path -- e.g. path_to_file = 'examples/molecule.xyz'. If None, then no output file is written.
    
    Returns
    -------
    str : xyz block
    """
    # 过滤掉无效的原子（原子序数 <= 0 或 > 118）
    valid_mask = (atomic_numbers > 0) & (atomic_numbers <= 118)
    valid_atomic_numbers = atomic_numbers[valid_mask]
    valid_positions = positions[valid_mask]
    
    N = valid_atomic_numbers.shape[0]
    
    if N == 0:
        raise ValueError("所有原子编号都无效（原子序数必须在 1-118 之间）")

    print("过滤掉无效的原子")

    xyz = ''
    xyz += f'{N}\n\n'
    for i in range(0, N):
        a = int(valid_atomic_numbers[i])
        p = valid_positions[i]
        symbol = rdkit.Chem.Atom(a).GetSymbol()
        if symbol == '*':
            # 跳过无效原子（理论上不应该出现，因为已经过滤了）
            continue
        xyz+= f'{symbol} {p[0]:>15.8f} {p[1]:>15.8f} {p[2]:>15.8f}\n'
    xyz+= '\n'

    if path_to_file is not None:
        with open(f'{path_to_file}', 'w') as f:
            f.write(xyz)
    return xyz


def get_xyz_content(atomic_numbers: np.ndarray,
                    positions: np.ndarray
                    ) -> str:
    """
    Get the xyz block of an atomistic structure.
    """
    xyz = write_xyz_file(atomic_numbers, positions, path_to_file=None)
    return xyz


def extract_mol_from_xyz_block(xyz_block: str,
                               charge: int = 0,
                               verbose: bool = False,
                               atoms: Optional[np.ndarray] = None,
                               positions: Optional[np.ndarray] = None,
                               bonds: Optional[np.ndarray] = None
                               ) -> rdkit.Chem.Mol:
    """
    Attempts to extract a mol object from an xyz block using build_3d_mol_from_arrays.

    Assumes that the xyz structure has hydrogens included explicitly.
    Requires bonds information - will return None if not provided.

    Arguments
    ---------
    xyz_block: str containing atomistic structure in xyz format.
    charge: int specifying the expected (overall) charge of the structure.
    verbose: bool indicating whether to print error statements upon extraction failure
    atoms: np.ndarray of atomic numbers. Required for bond determination.
    positions: np.ndarray of 3D positions. Required for bond determination.
    bonds: np.ndarray of bond types (edge list format). Required - will return None if not provided.

    Returns
    -------
    rdkit.Chem.rdchem.Mol object if successful, None otherwise
    """
    # 必须提供键信息
    if bonds is None or atoms is None or positions is None:
        if verbose:
            print('未提供键信息，无法构建分子')
        return None
    
    if verbose:
        print('使用 build_3d_mol_from_arrays 方法构建分子')
    
    try:
        # 过滤有效原子
        valid_atoms = []
        valid_positions = []
        valid_indices = []
        
        for i, (atom, pos) in enumerate(zip(atoms, positions)):
            atomic_number = int(atom)
            if atomic_number <= 0 or atomic_number > 118:
                continue
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                continue
            if np.any(np.abs(pos) > 1000):
                continue
                
            valid_atoms.append(atomic_number)
            valid_positions.append(pos)
            valid_indices.append(i)
        
        if len(valid_atoms) == 0:
            if verbose:
                print('过滤后没有有效原子')
            return None
        
        # 构建原始邻接矩阵
        num_atoms_original = len(atoms)
        expected_edges = num_atoms_original * (num_atoms_original - 1) // 2
        
        if len(bonds) != expected_edges:
            if verbose:
                print(f'键数据长度({len(bonds)})与期望边数({expected_edges})不匹配')
            return None
        
        # 创建上三角邻接矩阵的edge_index
        edge_sources = []
        edge_targets = []
        for i in range(num_atoms_original):
            for j in range(i + 1, num_atoms_original):
                edge_sources.append(i)
                edge_targets.append(j)
        
        edge_index = np.array([edge_sources, edge_targets])
        original_bond_adjacent_array = edge_list_to_adjacency_matrix(
            num_atoms_original, edge_index, bonds
        )
        
        # 提取有效原子之间的键
        num_valid = len(valid_atoms)
        bond_adjacent_array = np.zeros((num_valid, num_valid), dtype=int)
        
        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                orig_i = valid_indices[i]
                orig_j = valid_indices[j]
                bond_adjacent_array[i, j] = original_bond_adjacent_array[orig_i, orig_j]
        
        # 使用 build_3d_mol_from_arrays 构建分子
        mol = build_3d_mol_from_arrays(
            atom_type_array=np.array(valid_atoms),
            bond_adjacent_array=bond_adjacent_array,
            positions_3d=np.array(valid_positions)
        )
        
        if mol is None:
            if verbose:
                print('build_3d_mol_from_arrays 构建分子失败')
            return None
        
        if verbose:
            print('使用 build_3d_mol_from_arrays 成功构建分子')
            
    except Exception as e:
        if verbose:
            print(f'使用 build_3d_mol_from_arrays 失败: {e}')
        return None

    num_radicals = sum([a.GetNumRadicalElectrons() for a in mol.GetAtoms()])
    if num_radicals != 0:
        if verbose:
            print('Extracted molecule has radical electrons')
        return None

    mol.UpdatePropertyCache()
    rdkit.Chem.GetSymmSSSR(mol)

    if '.' in Chem.MolToSmiles(mol):
        if verbose:
            print('Mol object was extracted but contained multiple molecules')
        return None

    num_formal_chg = 0
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            num_formal_chg += 1
        if num_formal_chg > 6:
            return None

    return mol


def get_mol_from_atom_pos(atoms: np.ndarray,
                          positions: np.ndarray,
                          bonds: Optional[np.ndarray] = None
                          ) -> Tuple[Union[Chem.Mol, None], int, str]:
    """
    Try to get a RDKit mol object from atom and coordinate arrays.

    Arguments
    ---------
    atoms : np.ndarray (N,) of atomic numbers of the generated molecule or (N,M) one-hot
        encoding.
    positions : np.ndarray (N,3) of coordinates for the generated molecule's atoms.
    bonds : Optional np.ndarray of bond types (edge list format). If provided, will use build_3d_mol_from_arrays.

    Returns
    -------
    Tuple
        mol : Chem.Mol or None
        charge : int overall charge of molecule
        xyz_block : str
    """
    if len(atoms.shape) == 2:
        atomic_nums = np.argmin(np.abs(atoms - 1.0), axis = -1)
    else:
        atomic_nums = atoms
    xyz_block = write_xyz_file(atomic_nums, positions)

    for charge in [0, 1, -1, 2, -2]:
        try:
            mol = extract_mol_from_xyz_block(
                xyz_block=xyz_block, 
                charge=charge,
                atoms=atomic_nums,
                positions=positions,
                bonds=bonds
            )
        except Exception as e:
            mol = None

        if mol is not None:
            break
    else:
        charge = 0
    return mol, charge, xyz_block


def get_smiles_from_atom_pos(atoms: np.ndarray,
                             positions: np.ndarray
                             ) -> Union[str, None]:
    """
    Try to get a SMILES string from atom and coordinate arrays.

    Arguments
    ---------
    atoms : np.ndarray (N,) of atomic numbers of the generated molecule or (N,M) one-hot
        encoding.
    positions : np.ndarray (N,3) of coordinates for the generated molecule's atoms.

    Returns
    -------
    SMILES str or None
    """
    mol, _, _ = get_mol_from_atom_pos(atoms=atoms, positions=positions)
    smiles = None
    if mol is not None:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    return smiles



def load_npz_to_df(npz_path: Union[Path, str],
                   file_id: bool
                   ) -> pd.DataFrame:
    """
    Function to load a single npz file and return a dataframe with expanded zero-dimensional arrays.
    This works specifically for files generated by ConditionalEvalPipeline.
    """
    data = np.load(npz_path, allow_pickle=True)
    df_dict = {}

    # Find the first non-zero dimensional array length (assumed to be N_i)
    length = None
    for key, arr in data.items():
        
        if arr.ndim == 1 and len(arr) < 50:  # Non-zero dimensional array
            length = len(arr)
            break
    
    # Ensure we have a valid length for the file
    if length is None:
        raise ValueError(f"No 1D array found in {npz_path}")
    
    # Fill in the dictionary with arrays
    for key, arr in data.items():
        if key in ('ref_surf_resampling_scores', 'ref_surf_esp_resampling_scores', 'ref_mol_morgan_fp'):
            continue
        if arr.ndim == 0:  # Zero-dimensional array
            df_dict[key] = np.repeat(arr, length)  # Repeat value to match length N_i
        elif arr.ndim == 1 and len(arr) == length:  # 1D arrays with length N_i
            df_dict[key] = arr
        else:
            raise ValueError(f"Inconsistent array length for {key} in {npz_path}")
    
    if file_id is not None:
        df_dict['file_id'] = np.repeat(file_id, length)
    
    return pd.DataFrame(df_dict)


def collate_npz_files(npz_files: List[Union[str, Path]], 
                      include_file_id: bool
                      ) -> pd.DataFrame:
    """
    Function to collate all npz files into a single dataframe.

    Arguments
    ---------
    npz_files : list of file paths
    include_file_id : bool Whether to include a column called "file_id" that groups together
        rows that came from the same file.
    
    Returns
    -------
    pd.DataFrame : rows are each sample, columns are each property, and it repeats any 0d arrays.
    """
    dfs = []
    for i, npz_file in enumerate(npz_files):
        if include_file_id:
            df = load_npz_to_df(npz_file, file_id=i)
        else:
            df = load_npz_to_df(npz_file, file_id=None)
        dfs.append(df)
    
    # Concatenate all dataframes
    return pd.concat(dfs, ignore_index=True)
