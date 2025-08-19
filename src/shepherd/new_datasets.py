import open3d 
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atom_coords, 
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates

import rdkit
import numpy as np
import torch
import torch_geometric

from copy import deepcopy

# Discrete Diffusion import 
import torch.nn.functional as F

class MarginalUniformTransition:
    def __init__(self, x_marginals, e_marginals=None, y_classes=0):
        self.x_marginals = x_marginals 
    def get_Qt_bar(self, alpha_t_bar, device):                
        # Simplified transition for demonstration
        K = self.x_marginals.shape[0]
        return PredefinedNoiseScheduleDiscrete.get_Qt_bar_from_marginals(self.x_marginals, alpha_t_bar, device)

class PredefinedNoiseScheduleDiscrete:
    def __init__(self, timesteps):
        self.T = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def __call__(self, t_normalized):
        t_int = (t_normalized * self.T).long()
        # 添加边界检查，防止索引超出范围
        t_int = torch.clamp(t_int, 0, self.T - 1)
        return self.betas[t_int]
    
    def get_alpha_bar(self, t_normalized):
        t_int = (t_normalized * self.T).long()
        # 添加边界检查，防止索引超出范围
        t_int = torch.clamp(t_int, 0, self.T - 1)
        return self.alpha_bars[t_int]
        
    @staticmethod
    def get_Qt_bar_from_marginals(mu_x, alpha_bar_t, device):
        """
        Same as in the original code.
        """
        mu_x = mu_x.to(device)
        alpha_bar_t = alpha_bar_t.to(device)
        K = mu_x.size(0)
        
        # Reshape alpha_bar_t and mu_x for broadcasting
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)  # (bs, 1, 1)
        mu_x = mu_x.view(1, 1, K)  # (1, 1, K)

        # Calculate the transition matrix Qt_bar
        Qt_bar = alpha_bar_t * torch.eye(K, device=device).unsqueeze(0) + (1. - alpha_bar_t) * mu_x
        
        return Qt_bar

# simple realization of DiscDiff
class DiscreteFeatureDiffusion:
    def __init__(self, timesteps, marginals):
        """
        A helper class to apply discrete diffusion noise to one-hot encoded features.
        
        Args:
            timesteps (int): Total number of diffusion timesteps (T).
            marginals (torch.Tensor): A 1D tensor with the marginal probability of each class.
        """
        self.T = timesteps
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=self.T)
        
        # Ensure marginals sum to 1
        marginals = marginals.float()
        feature_marginals = marginals / torch.sum(marginals)
        
        print(f"Initialized DiscreteFeatureDiffusion with marginals: {feature_marginals}")
        
        # We only need the transition model for features (X/E), not for properties (y)
        self.transition_model = MarginalUniformTransition(x_marginals=feature_marginals, e_marginals=None, y_classes=0)

    def get_params_for_t(self, t_int, device):
        """ Get diffusion parameters for a given integer timestep t. """
        if t_int.numel() > 1:
            t_int = t_int[0]

        t_float = t_int.float() / self.T
        s_int = t_int - 1
        s_float = torch.clamp(s_int.float(), min=-1.0) / self.T
        
        beta_t = self.noise_schedule(t_normalized=t_float)
        if (s_int < 0).any():
             alpha_s_bar = torch.ones_like(beta_t)
        else:
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        
        return {
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar
        }

    def apply_noise(self, features_0, t_int, device):
        """
        Sample x_t given x_0 and t.

        Args:
            features_0 (torch.Tensor): The original one-hot encoded features [N, C].
            t_int (int): A single integer timestep.
        
        Returns:
            A dictionary containing:
            - 'features_t': The noised one-hot features [N, C].
            - 'params': A dict with beta_t, alpha_s_bar, alpha_t_bar for loss calculation.
        """
        # Get transition parameters
        t_int_tensor = torch.tensor([t_int], device=device)
        params = self.get_params_for_t(t_int_tensor, device)
        alpha_t_bar = params['alpha_t_bar']
        
        # Get the transition matrix Q_t_bar
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=device)
        
        prob_t = torch.matmul(features_0, Qtb.squeeze(0)) # [N, C]
        
        # ========================= ROBUSTNESS MODIFICATION START =========================
        # 对概率进行防御性处理，防止 torch.multinomial 崩溃

        # 1. 确保概率值非负 (防止浮点数精度问题)
        prob_t = torch.clamp(prob_t, min=0)
        
        # 2. 处理行和为零的情况
        row_sums = prob_t.sum(dim=-1, keepdim=True)
        
        # 3. 对于行和接近于零的行，替换为一个均匀分布，避免除以零
        #    对于行和大于零的行，进行归一化，确保和为 1
        uniform_fallback = torch.ones_like(prob_t) / prob_t.shape[-1]
        prob_t = torch.where(row_sums > 1e-6, prob_t / row_sums, uniform_fallback)
        
        # ========================= ROBUSTNESS MODIFICATION END ===========================
        
        # Sample from the categorical distribution to get the noised features
        sampled_indices_t = torch.multinomial(prob_t, num_samples=1).squeeze(-1) # [N]
        features_t = F.one_hot(sampled_indices_t, num_classes=features_0.shape[1]).float() # [N, C]
        
        return features_t

# in most cases, this function won't be used, as we use xTB charges rather than MMFF charges.
def get_atomic_partial_charges(mol: rdkit.Chem.Mol) -> np.ndarray:
    """
    Gets partial charges for a given molecule.
    Assumes the input "mol" already has an optimized conformer. Gets partial charges from
    MMFF or Gasteiger.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object
        RDKit molecule object with an optimized geometry in conformers.
    
    Returns
    -------
    np.ndarray (N)
        Partial charges for each atom in the molecule.
    """
    
    try:
        mol.GetConformer()
    except ValueError as e:
        raise ValueError(f"Provided rdkit.Chem.Mol object did not have conformer embedded.", e)
    
    molec_props = rdkit.Chem.AllChem.MMFFGetMoleculeProperties(mol)
    if molec_props:
        # electron units
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(mol.GetAtoms())])
    else:
        print("MMFF charges not available for the input molecule, defaulting to Gasteiger charges.")
        rdkit.Chem.AllChem.ComputeGasteigerCharges(mol)
        charges=np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    
    return charges


class HeteroDataset(torch_geometric.data.Dataset):
    def __init__(self, 
                 
            molblocks_and_charges, 
            
            noise_schedule_dict,
            
            # NEW: Parameters for discrete diffusion
            atom_marginals_x1,
            bond_marginals_x1,
            pharm_marginals_x4,

            explicit_hydrogens = True,
            use_MMFF94_charges = False,
            formal_charge_diffusion = False,
            
            x1 = True,
            x2 = True,
            x3 = True,
            x4 = True,
                
            recenter_x1 = True, 
            add_virtual_node_x1 = True, 
            remove_noise_COM_x1 = True,
            atom_types_x1 = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si'],
            charge_types_x1 = [0,1,2,-1,-2],
            bond_types_x1 = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
            scale_atom_features_x1 = 1.0,
            scale_bond_features_x1 = 1.0,

            
            
            independent_timesteps_x2 = False,
            recenter_x2 = False, # we want the center of x2 to be the virtual node (whose position is the center of x1)
            add_virtual_node_x2 = True,
            remove_noise_COM_x2 = False,
            num_points_x2 = 75,

            
            independent_timesteps_x3 = False,
            recenter_x3 = False,
            add_virtual_node_x3 = True,
            remove_noise_COM_x3 = False,
            num_points_x3 = 75,
            scale_node_features_x3 = 1.0,
            
                 
            independent_timesteps_x4 = False,
            recenter_x4 = False, 
            add_virtual_node_x4 = True, # must be true, for edge-case where molecule doesn't have any pharamcophores
            remove_noise_COM_x4 = False,
            max_node_types_x4 = 16, # number of pharmacophore types (can be set larger than represented in dataset)
            scale_node_features_x4 = 1.0,
            scale_vector_features_x4 = 1.0,
            multivectors = False,
            check_accessibility = False,
            
            probe_radius = 0.6,                 
        ):
        
        super().__init__() # Important to call parent constructor

        self.molblocks_and_charges = molblocks_and_charges
        self.length = len(molblocks_and_charges)
        self.use_MMFF94_charges = use_MMFF94_charges
        
        self.noise_schedule_dict = noise_schedule_dict
        
        self.explicit_hydrogens = explicit_hydrogens
        assert self.explicit_hydrogens == True
        
        self.formal_charge_diffusion = formal_charge_diffusion
        
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        
        self.recenter_x1 = recenter_x1
        self.add_virtual_node_x1 = add_virtual_node_x1
        self.remove_noise_COM_x1 = remove_noise_COM_x1 # True
        self.atom_types_x1 = atom_types_x1
        self.charge_types_x1 = charge_types_x1
        self.bond_types_x1 = bond_types_x1
        self.scale_atom_features_x1 = scale_atom_features_x1
        self.scale_bond_features_x1 = scale_bond_features_x1
        
        self.recenter_x2 = recenter_x2 
        self.add_virtual_node_x2 = add_virtual_node_x2
        self.remove_noise_COM_x2 = remove_noise_COM_x2
        self.num_points_x2 = num_points_x2
        self.independent_timesteps_x2 = independent_timesteps_x2
        
        self.recenter_x3 = recenter_x3
        self.add_virtual_node_x3 = add_virtual_node_x3
        self.remove_noise_COM_x3 = remove_noise_COM_x3
        self.num_points_x3 = num_points_x3
        self.independent_timesteps_x3 = independent_timesteps_x3
        self.scale_node_features_x3 = scale_node_features_x3

        self.independent_timesteps_x4 = independent_timesteps_x4
        self.recenter_x4 = recenter_x4
        self.add_virtual_node_x4 = add_virtual_node_x4 
        self.remove_noise_COM_x4 = remove_noise_COM_x4
        self.max_node_types_x4 = max_node_types_x4
        self.scale_node_features_x4 = scale_node_features_x4
        self.scale_vector_features_x4 = scale_vector_features_x4
        self.multivectors = multivectors
        self.check_accessibility = check_accessibility
        
        self.probe_radius = probe_radius
        self.scale_electrostatics = self.scale_node_features_x3  # alias
    
        # NEW: Initialize discrete diffusion helpers
        
        T = len(noise_schedule_dict['x1']['ts'])
        if self.x1:
            self.x1_atom_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=atom_marginals_x1)
            self.x1_bond_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=bond_marginals_x1)
        if self.x4:
            self.x4_pharm_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=pharm_marginals_x4)
    
    def get_x1_data(self, mol, t, alpha_dash_t, sigma_dash_t):
        # this uses the same noise schedule for both positions and atom types/features
        
        data = {}
        data['timestep'] = torch.tensor([t], dtype=torch.long)

        atom_types = [self.atom_types_x1.index(a.GetSymbol()) for a in mol.GetAtoms()]
        if self.formal_charge_diffusion:
            formal_charges = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
            formal_charge_map = {c:self.charge_types_x1.index(c) for c in self.charge_types_x1}
            formal_charges_mapped = [formal_charge_map[f] for f in formal_charges]
            
        pos = np.array(mol.GetConformer().GetPositions())
        num_atoms = len(pos)
        
        bond_adj = 1-np.diag(np.ones(num_atoms, dtype = int))
        bond_adj = np.triu(bond_adj) # directed graph, to only include 1 edge per bond
        bond_edge_index = np.stack(bond_adj.nonzero(), axis = 0) # this doesn't include any edges to the virtual node
        bond_types_dict = {b:self.bond_types_x1.index(b) for b in self.bond_types_x1}
        max_bond_types_x1 = len(bond_types_dict)
        bond_types = []
        for b in range(bond_edge_index.shape[1]):
            idx_1 = int(bond_edge_index[0, b])
            idx_2 = int(bond_edge_index[1, b])
            bond = mol.GetBondBetweenAtoms(idx_1, idx_2)
            if bond is None:
                bond_types.append(bond_types_dict[None]) # non-bonded edge type; == 0
            else:
                bond_type = bond_types_dict[str(bond.GetBondType())]
                bond_types.append(bond_type)
        data['bond_edge_mask'] = torch.from_numpy((np.array(bond_types) != 0).copy()).bool() # True indicates a real bond
        
        
        COM_before_centering = pos.mean(0)[None, ...]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)
        if self.recenter_x1:
            pos = pos_recentered
        COM = pos.mean(0)[None, ...]
        data['com'] = torch.from_numpy(COM.copy()).float()

        virtual_node_mask = np.zeros(pos.shape[0] + int(self.add_virtual_node_x1))
        if self.add_virtual_node_x1: # should change according to desired behavior
            assert self.atom_types_x1[0] == None
            atom_types.insert(0, 0)
            bond_edge_index = bond_edge_index + 1 # accounting for virtual node
            virtual_node_pos = COM
            pos = np.concatenate([virtual_node_pos, pos], axis = 0) # setting virtual node position to (non-zero) COM
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0) # setting virtual node position to zero
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1
        num_nodes = num_atoms + int(self.add_virtual_node_x1)
        
        data['bond_edge_index'] = torch.from_numpy(bond_edge_index.copy()).long()
        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()
        
        
        # (scaled) one-hot embedding of atom types and formal charges for non-noised structure
        x = np.zeros((num_nodes, len(self.atom_types_x1))) #torch.tensor(atomic_numbers, dtype = torch.long)
        x[np.arange(num_nodes), atom_types] = 1
        x = x * self.scale_atom_features_x1
        if self.formal_charge_diffusion:
            x_formal_charges = np.zeros((len(formal_charges_mapped), len(self.charge_types_x1)))
            x_formal_charges[np.arange(len(formal_charges_mapped)), formal_charges_mapped] = 1
            x_formal_charges = x_formal_charges * self.scale_atom_features_x1
            if self.add_virtual_node_x1:
                # virtual node has all zeros for the formal charge one-hot features
                x_formal_charges = np.concatenate((np.zeros(len(self.charge_types_x1), dtype = x_formal_charges.dtype)[None, ...], x_formal_charges), axis = 0)
            x = np.concatenate((x, x_formal_charges), axis = 1)
        data['x'] = torch.from_numpy(x.copy()).float()
        
        
        # (scaled) one-hot embedding of bond types for non-noised structure
            # this doesn't include any edges to the virtual node
        bond_edge_x = np.zeros((bond_edge_index.shape[1], max_bond_types_x1))
        bond_edge_x[np.arange(len(bond_types)), bond_types] = 1
        bond_edge_x = bond_edge_x * self.scale_bond_features_x1
        data['bond_edge_x'] = torch.from_numpy(bond_edge_x.copy()).float()
        
        # ========================= MODIFICATION START =========================
        
        # --- 1. 保存 t=0 的干净特征 ---
        # 这是我们最重要的修改！损失函数需要这些 t=0 的真实标签。
        data['x_0'] = data['x'].clone()
        data['bond_edge_x_0'] = data['bond_edge_x'].clone()

        # --- 2. 处理连续特征 (坐标) 的加噪 (逻辑保持不变) ---
        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0
        if self.remove_noise_COM_x1:
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis = 0) 
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()
        
        pos_forward_noised = alpha_dash_t * pos  +  sigma_dash_t * pos_noise 
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()

        # --- 3. 处理离散特征 (原子 & 键类型) 的加噪 ---
        # 我们将完全替换旧的加噪逻辑
        device = data['x'].device # 获取张量所在的设备
        t_tensor = torch.tensor([t], device=device)

        # 对原子类型加噪 (只对非虚拟节点操作)
        x_clean_no_vn = data['x_0'][~virtual_node_mask]
        x_noised_no_vn = self.x1_atom_diffuser.apply_noise(x_clean_no_vn, t_tensor, device)
        
        x_forward_noised = data['x_0'].clone() # 从干净的 t=0 数据开始
        x_forward_noised[~virtual_node_mask] = x_noised_no_vn
        data['x_forward_noised'] = x_forward_noised
        
        # 对键类型加噪
        bond_edge_x_clean = data['bond_edge_x_0']
        bond_edge_x_noised = self.x1_bond_diffuser.apply_noise(bond_edge_x_clean, t_tensor, device)
        data['bond_edge_x_forward_noised'] = bond_edge_x_noised

        # 旧的 x_noise 和 bond_edge_x_noise 不再需要，可以移除或设为零
        data['x_noise'] = torch.zeros_like(data['x'])
        data['bond_edge_x_noise'] = torch.zeros_like(data['bond_edge_x'])

        # ========================= MODIFICATION END ===========================

        return data, pos, virtual_node_mask
    
    
    # 根据原子半径 (radii) 和中心坐标 (atom_centers) 生成一个分子表面点云，然后构造条件输入数据结构 data，包含坐标、中心信息、是否有虚拟节点等。

    def get_x2_data(self, radii, atom_centers, num_points, recenter, add_virtual_node, remove_noise_COM, t, alpha_dash_t, sigma_dash_t, virtual_node_pos = None):
        
        data = {}
        data['timestep'] = torch.tensor([t], dtype=torch.long)
        
        pos = get_molecular_surface(
            atom_centers,
            radii,
            num_points=num_points,
            probe_radius = self.probe_radius, # 探针半径
            num_samples_per_atom = 20,
        )
        
        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()

        # 中心化处理
        pos_recentered = pos - pos.mean(0)
        if recenter:
            pos = pos_recentered

        COM = pos.mean(0)[None, :]
        data['com'] = torch.from_numpy(COM.copy()).float()
        
        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()        
        
        # one-hot embedding indicating real vs virtual nodes
        x = np.zeros((pos.shape[0], 2))
        data['x'] = torch.from_numpy(x.copy()).float()

        # 作为条件不加噪

        data['x_forward_noised'] = data['x'].clone() 
    
        data['pos_noise'] = torch.zeros_like(data['pos'])
        
        data['pos_forward_noised'] = data['pos'].clone()
        
        return data, pos, virtual_node_mask
    
    
    # 不加噪音，条件化
    def get_x3_data_electrostatics_only(self, charges, charge_centers, data, pos, virtual_node_mask, t, alpha_dash_t, sigma_dash_t):
        
        x = get_electrostatics_given_point_charges(charges, charge_centers, pos) # compute ESP at each point in pos
        x = x * self.scale_node_features_x3
                
        data['x'] = torch.from_numpy(x.copy()).float()
        
        data['x_noise'] = torch.zeros_like(data['x'])
        
        data['x_forward_noised'] = data['x'].clone()
        
        return data
    

    def get_x4_data(self, mol, recenter, add_virtual_node, remove_noise_COM, t, alpha_dash_t, sigma_dash_t, virtual_node_pos = None):
        
        # it is important to include a virtual node in case there are NO pharmacophores in the molecule
        assert add_virtual_node

        data = {}
        data['timestep'] = torch.tensor([t], dtype=torch.long)
        
        # 1. --- 获取基础药效团信息 ---
        pharm_types, pos, direction = get_pharmacophores(
            mol, 
            multi_vector = self.multivectors, 
            check_access=self.check_accessibility,
        )
        # 为虚拟节点在 0 号位腾出空间
        pharm_types = pharm_types + 1 
        # 给坐标加一点微小的扰动，防止完全重合
        pos = pos + np.random.randn(*pos.shape) * 0.05
        
        # 2. --- 统一处理虚拟节点和空分子的情况 ---
        # 如果没有药效团，就创建一个占位的
        if pharm_types.shape[0] == 0:
            pharm_types = np.array([]) # 保持为空，下面会统一处理
            pos = np.empty((0, 3))
            direction = np.empty((0, 3))
            
        # 添加虚拟节点
        # 它的类型是 0
        pharm_types = np.concatenate([np.array([0]), pharm_types], axis=0)
        
        # 计算虚拟节点的位置 (如果未提供)
        if virtual_node_pos is None:
            if pos.shape[0] > 1: # 如果除了虚拟节点还有其他点
                virtual_node_pos = pos[1:, :].mean(0, keepdims=True)
            else: # 如果只有虚拟节点
                virtual_node_pos = np.zeros((1, 3))

        pos = np.concatenate([virtual_node_pos, pos], axis=0)
        direction = np.concatenate([np.zeros((1, 3)), direction], axis=0)
        
        # 3. --- 计算坐标、掩码和 t=0 特征 ---
        COM_before_centering = pos.mean(0, keepdims=True)
        pos_recentered = pos - COM_before_centering
        if recenter:
            pos = pos_recentered
        
        COM = pos.mean(0, keepdims=True)
        virtual_node_mask = np.zeros(pos.shape[0], dtype=bool)
        virtual_node_mask[0] = True

        # 创建 t=0 的离散特征 x 和 x_0
        x = np.zeros((pharm_types.size, self.max_node_types_x4))
        x[np.arange(pharm_types.size), pharm_types] = 1
        data['x'] = torch.from_numpy(x.copy()).float() * self.scale_node_features_x4
        data['x_0'] = data['x'].clone() # <--- 关键！

        # 存储其他 t=0 的属性
        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['direction'] = torch.from_numpy(direction.copy()).float() * self.scale_vector_features_x4
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask)
        
        # 4. --- 对所有特征进行加噪 ---
        # 连续特征 (坐标和方向)
        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0 # 虚拟节点不加噪
        if remove_noise_COM:
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - pos_noise[~virtual_node_mask].mean(0)
        
        direction_noise = np.random.randn(*direction.shape)
        direction_noise[virtual_node_mask] = 0.0 # 虚拟节点不加噪

        # 存储噪声本身
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()
        data['direction_noise'] = torch.from_numpy(direction_noise.copy()).float()

        # 应用噪声
        data['pos_forward_noised'] = data['pos'] + data['pos_noise'] * sigma_dash_t
        data['direction_forward_noised'] = data['direction'] + data['direction_noise'] * sigma_dash_t
        
        # 离散特征 (药效团类型)
        device = data['x'].device
        t_tensor = torch.tensor([t], device=device).long()
        
        # 只对非虚拟节点加噪
        x_clean_no_vn = data['x_0'][~virtual_node_mask]
        
        if x_clean_no_vn.shape[0] > 0: # 确保有真实节点才加噪
            x_noised_no_vn = self.x4_pharm_diffuser.apply_noise(x_clean_no_vn, t_tensor, device)
            x_forward_noised = data['x_0'].clone()
            x_forward_noised[~virtual_node_mask] = x_noised_no_vn
        else: # 如果只有虚拟节点，则不加噪
            x_forward_noised = data['x_0'].clone()
            
        data['x_forward_noised'] = x_forward_noised
        # 占位符，因为损失函数不再需要它
        data['x_noise'] = torch.zeros_like(data['x'])

        return data
    
    def __getitem__(self, k):
        
        mol_block = self.molblocks_and_charges[k][0]
        charges = np.array(self.molblocks_and_charges[k][1]) # precomputed charges (e.g., from xTB)
        
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs = False) # 表示不移除氢原子
        atomic_numbers = np.array([int(a.GetAtomicNum()) for a in mol.GetAtoms()]) # 获取分子中每个原子的原子序数
        
        # 表示在分子结构中明确地包含氢原子
        assert self.explicit_hydrogens # if we want to treat hydrogens implicitly, then we need to adjust how x2,x3,x4 are computed
        
        # centering molecule coordinates
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)

        # mol = update_mol_coordinates(mol, mol_coordinates, copy = False)
        mol = update_mol_coordinates(mol, mol_coordinates)

        # 获取每个原子的范德华半径
        radii = get_atomic_vdw_radii(mol)

        # MMFF94 方法计算部分电荷
        if self.use_MMFF94_charges:
            charges = get_atomic_partial_charges(mol) #MMFF charges
        
        data_dict = {
            'molecule_id': torch.tensor([k], dtype=torch.long),
            'x1': {},
            'x2': {},
            'x3': {},
            'x4': {},
        }
        
        
        if self.x1:
            ts = self.noise_schedule_dict['x1']['ts']
            
            # t = np.random.choice(ts)  
            # random time step sampled uniformly from time sequence
            T = ts.shape[0]

            ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
            ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
            ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
            
            ts_prob = np.random.uniform(0,1)

            if ts_prob < 0.075:
                t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
            elif ts_prob < (0.075 + 0.75):
                t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
            else:
                t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
            
            ts_x1 = ts
            t_x1 = t
            t_idx = np.where(ts == t)[0][0]

            alpha_t = self.noise_schedule_dict['x1']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x1']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x1']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x1']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x1']['sigma_dash_ts'][t_idx]
            
            x1_data, x1_pos, x1_virtual_node_mask = self.get_x1_data(mol, t, alpha_dash_t, sigma_dash_t)
            
            x1_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x1_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x1_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x1_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x1'] = x1_data
        
        
        if self.x2:
            
            if self.independent_timesteps_x2:
                ts = self.noise_schedule_dict['x2']['ts']
                
                #t = np.random.choice(ts)  # random time step sampled uniformly from time sequence
                T = ts.shape[0]
                ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
                ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
                ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
                else:
                    t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
                
            else:
                assert self.x1 == True
                # use same time sequence as x1
                assert (self.noise_schedule_dict['x2']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1
                t = t_x1
            ts_x2 = ts
            t_x2 = t
            t_idx = np.where(ts == t)[0][0]
            alpha_t = self.noise_schedule_dict['x2']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x2']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x2']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x2']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x2']['sigma_dash_ts'][t_idx]
            
            if self.x1:
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x2) and (self.recenter_x2 == False)) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None # this will get re-set to be the COM of x2 (NOT mol_coordinates) in get_x2_data
            
            x2_data, x2_pos, x2_virtual_node_mask = self.get_x2_data(
                radii,
                atom_centers, 
                self.num_points_x2,
                self.recenter_x2,
                self.add_virtual_node_x2,
                self.remove_noise_COM_x2,
                t, alpha_dash_t, sigma_dash_t, 
                virtual_node_pos = virtual_node_pos,
            )
            
            x2_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x2_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x2_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x2_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x2'] = x2_data
        
        
        
        if self.x3:
            if self.independent_timesteps_x3:
                ts = self.noise_schedule_dict['x3']['ts']
                
                #t = np.random.choice(ts)  # random time step sampled uniformly from time sequence
                T = ts.shape[0]
                ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
                ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
                ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
                else:
                    t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
                
            else:
                assert self.x1 == True
                
                # use same time sequence as x1 
                assert (self.noise_schedule_dict['x3']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1
                t = t_x1
            
            ts_x3 = ts
            t_x3 = t
            t_idx = np.where(ts == t)[0][0]
            alpha_t = self.noise_schedule_dict['x3']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x3']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x3']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x3']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x3']['sigma_dash_ts'][t_idx]
            
            if self.x1:
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x3) and (self.recenter_x3 == False)) else None  
            else:
                atom_centers = mol_coordinates # this might need to be centered before we assign it to charge_centers
                virtual_node_pos = None # this will get re-set to be the COM of x3 (NOT mol_coordinates) in get_x3_data
            
            # we use the same surface cloud formulation as x2 for the points in x3
            x3_data, x3_pos, x3_virtual_node_mask = self.get_x2_data(
                radii, 
                atom_centers, 
                self.num_points_x3, 
                self.recenter_x3, 
                self.add_virtual_node_x3, 
                self.remove_noise_COM_x3, 
                t, alpha_dash_t, sigma_dash_t, 
                virtual_node_pos = virtual_node_pos,
            )
            
            # the x3 point cloud, if re-centered, is displaced from the atom centers used to generate it. 
                # Before computing electrostatics for x3, we have to displace the charge centers to account for this.
            x3_COM_displacement = (x3_data['com'] - x3_data['com_before_centering']).detach().cpu().numpy()
            charge_centers = atom_centers + x3_COM_displacement
            
            # same noise is applied to both coordinates and features
            x3_data = self.get_x3_data_electrostatics_only(
                charges, 
                charge_centers, 
                x3_data, 
                x3_pos, 
                x3_virtual_node_mask, 
                t, alpha_dash_t, sigma_dash_t,
            )
            
            x3_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x3_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x3_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x3_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x3'] = x3_data

        
        if self.x4:
            
            if self.independent_timesteps_x4:
                ts = self.noise_schedule_dict['x4']['ts']
                
                #t = np.random.choice(ts)  # random time step sampled uniformly from time sequence
                T = ts.shape[0]
                ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
                ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
                ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
                else:
                    t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
                
            else:
                assert self.x1 == True
                # use same time sequence as x1
                assert (self.noise_schedule_dict['x4']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1
                t = t_x1
            ts_x4 = ts
            t_x4 = t
            t_idx = np.where(ts == t)[0][0]
            alpha_t = self.noise_schedule_dict['x4']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x4']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x4']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x4']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x4']['sigma_dash_ts'][t_idx]
            
            if self.x1:
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x4) and (self.recenter_x4 == False)) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None # this will get re-set to be the COM of x4 (NOT mol_coordinates) in get_x4_data
            
            x4_data = self.get_x4_data(
                mol, 
                self.recenter_x4, 
                self.add_virtual_node_x4, 
                self.remove_noise_COM_x4, 
                t, 
                alpha_dash_t,
                sigma_dash_t,
                virtual_node_pos,
            )
            
            x4_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x4_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x4_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x4_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x4'] = x4_data
        
        
        data = torch_geometric.data.HeteroData()
        if 'molecule_id' in data_dict:
            data.molecule_id = data_dict['molecule_id']

        if 'x1' in data_dict and data_dict['x1']:
            x1_data = data_dict['x1']
            
            node_keys = ['pos', 'pos_recentered', 'pos_forward_noised', 'pos_noise',
                'x', 'x_0', 'x_forward_noised', 'x_noise',
                'virtual_node_mask', 'com', 'com_before_centering',
                'timestep', 'alpha_t', 'sigma_t', 'alpha_dash_t', 'sigma_dash_t']

            # Separate continuous, discrete, and edge data
            for key in node_keys:
                if key in x1_data:
                    data['x1'][key] = x1_data[key]

            # 将所有边相关的属性存入 data['x1', 'bond', 'x1']
            edge_keys = ['bond_edge_index', 'bond_edge_mask',
                         'bond_edge_x', 'bond_edge_x_0', 'bond_edge_x_forward_noised', 'bond_edge_x_noise']
            
            data['x1', 'bond', 'x1'].edge_index = x1_data['bond_edge_index']
            data['x1', 'bond', 'x1'].mask = x1_data['bond_edge_mask']
            # 使用 .x 作为标准属性名
            data['x1', 'bond', 'x1'].x = x1_data['bond_edge_x']
            data['x1', 'bond', 'x1'].x_0 = x1_data['bond_edge_x_0']
            data['x1', 'bond', 'x1'].x_forward_noised = x1_data['bond_edge_x_forward_noised']
            data['x1', 'bond', 'x1'].x_noise = x1_data['bond_edge_x_noise']

        if 'x2' in data_dict and data_dict['x2']:
            for key, value in data_dict['x2'].items():
                data['x2'][key] = value
        if 'x3' in data_dict and data_dict['x3']:
            for key, value in data_dict['x3'].items():
                data['x3'][key] = value


        if 'x4' in data_dict and data_dict['x4']:

            x4_data = data_dict['x4']
            for key, value in x4_data.items():
                data['x4'][key] = value

            # # PAST

            # for key, value in data_dict['x4'].items():
            #     data['x4'][key] = value

        return data
    
    
    # for compatibility with other PyG versions
    def __len__(self): return self.length
    def len(self): return self.__len__()
    def getitem(self, k): return self.__getitem__(k)
    def get(self, k): return self.__getitem__(k)
