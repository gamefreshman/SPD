import numpy as np
import torch.nn.functional as F
import torch
import torch_geometric
import torch_scatter
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from shepherd.model.model import Model

class LightningModule(pl.LightningModule):
    
    def __init__(self, params):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.params = params
        
        self.model = Model(params)
        
        self.train_x1_denoising = params['training']['train_x1_denoising']
        self.train_x2_denoising = params['training']['train_x2_denoising']
        self.train_x3_denoising = params['training']['train_x3_denoising']
        self.train_x4_denoising = params['training']['train_x4_denoising']
        
        self.lr = params['training']['lr']
        self.min_lr = params['training']['min_lr']
        self.lr_steps = params['training']['lr_steps']
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
        # exponential lr decay from self.lr to self.min_lr in self.lr_steps steps
        gamma = (self.min_lr / self.lr) ** (1.0 / self.lr_steps)
        func = lambda step: max(gamma**(step), self.min_lr / self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
            "name": None,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
    
    def get_training_input_dict(self, data):
        
        input_dict = {}
        
        if self.params['dataset']['compute_x1']:
            # ========================= MODIFICATION START =========================
            
            # 1. 获取节点存储
            x1_data = data['x1']
            
            # 2. 获取边存储
            # 新版本的 PyG 使用 data[('x1', 'bond', 'x1')]
            # 为了兼容性，我们可以用 try-except
            try:
                edge_store = data['x1', 'bond', 'x1']
            except KeyError:
                edge_store = data[('x1', 'bond', 'x1')]

            # 3. 从正确的位置获取边属性
            bond_edge_mask = edge_store.mask
            bond_edge_index = edge_store.edge_index
            bond_edge_x = edge_store.x_forward_noised
            bond_edge_x_noise = edge_store.x_noise
            # 获取我们新加的 t=0 标签
            true_bond_types_t0 = edge_store.x_0 

            input_dict['x1'] = {
                'decoder': {
                    'pos': x1_data.pos_forward_noised,
                    'x': x1_data.x_forward_noised,
                    'batch': x1_data.batch,
                    'timestep': x1_data.timestep,
                    'alpha_t': x1_data.alpha_t,
                    'sigma_t': x1_data.sigma_t,
                    'alpha_dash_t': x1_data.alpha_dash_t,
                    'sigma_dash_t': x1_data.sigma_dash_t,
                    'virtual_node_mask': x1_data.virtual_node_mask,
                    'pos_noise': x1_data.pos_noise,
                    'x_noise': x1_data.x_noise,
                    # 获取 t=0 的原子标签
                    'true_atom_types_t0': x1_data.x_0,

                    # 添加边相关的信息
                    'bond_edge_mask': bond_edge_mask,
                    'bond_edge_index': bond_edge_index,
                    'bond_edge_x': bond_edge_x,
                    'bond_edge_x_noise': bond_edge_x_noise,
                    'true_bond_types_t0': true_bond_types_t0,
                },
            }
            # 形式电荷扩散, 还未实现
            # if self.params['x1_formal_charge_diffusion']:
            #     input_dict['x1']['decoder']['formal_charges'] = x1_data.formal_charges
            #     input_dict['x1']['decoder']['formal_charges_0'] = x1_data.formal_charges_0
            #     input_dict['x1']['decoder']['formal_charges_forward_noised'] = x1_data.formal_charges_forward_noised

            # ========================= MODIFICATION END ===========================
        
        
        if self.params['dataset']['compute_x2']:
            input_dict['x2'] =  {
                
                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'pos': data['x2'].pos_forward_noised, # this is the structure after forward-noising
                    'x': data['x2'].x_forward_noised, # currently, this is just one-hot embedding of virtual / real node (equal to data['x2'].x)
                    'batch': data['x2'].batch,
                    
                    'timestep': data['x2'].timestep,
                    'alpha_t': data['x2'].alpha_t,
                    'sigma_t': data['x2'].sigma_t,
                    'alpha_dash_t': data['x2'].alpha_dash_t,
                    'sigma_dash_t': data['x2'].sigma_dash_t,
                    
                    'virtual_node_mask': data['x2'].virtual_node_mask,
                    
                    'pos_noise': data['x2'].pos_noise, # this is the added (gaussian) noise
                    
                },
            }
        
        
        if self.params['dataset']['compute_x3']:
            input_dict['x3'] = {
                
                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'pos': data['x3'].pos_forward_noised, # this is the structure after forward-noising
                    'x': data['x3'].x_forward_noised, # this is the structure after forward-noising
                    'batch': data['x3'].batch,
                    
                    'timestep': data['x3'].timestep,
                    'alpha_t': data['x3'].alpha_t,
                    'sigma_t': data['x3'].sigma_t,
                    'alpha_dash_t': data['x3'].alpha_dash_t,
                    'sigma_dash_t': data['x3'].sigma_dash_t,
                    
                    'virtual_node_mask': data['x3'].virtual_node_mask,
                    
                    'pos_noise': data['x3'].pos_noise, # this is the added (gaussian) noise
                    'x_noise': data['x3'].x_noise, # this is the added (gaussian) noise
                    
                },
            }
        
        
        if self.params['dataset']['compute_x4']:
            x4_data_store = data['x4']
            
            input_dict['x4'] = {
                'decoder': {
                    'x': x4_data_store.x_forward_noised,
                    'pos': x4_data_store.pos_forward_noised,
                    'direction': x4_data_store.direction_forward_noised,
                    'batch': x4_data_store.batch,
                    
                    'timestep': x4_data_store.timestep,
                    'alpha_t': x4_data_store.alpha_t,
                    'sigma_t': x4_data_store.sigma_t,
                    'alpha_dash_t': x4_data_store.alpha_dash_t,
                    'sigma_dash_t': x4_data_store.sigma_dash_t,
                    
                    'virtual_node_mask': x4_data_store.virtual_node_mask,
                    
                    'direction_noise': x4_data_store.direction_noise,
                    'pos_noise': x4_data_store.pos_noise,
                    'x_noise': x4_data_store.x_noise,
                    
                    'true_pharm_types_t0': x4_data_store.x_0
                },
            }

        input_dict['device'] = self.device
        input_dict['dtype'] = torch.float32
        return input_dict
    
    
    def forward_training(self, input_dict):
        _, output_dict = self.model.forward(input_dict)
        return output_dict
    
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch
        batch_size = data.molecule_id.shape[0]
        
        input_dict = self.get_training_input_dict(data)
        
        output_dict = self.forward_training(input_dict)
        
        loss = 0.0
        #loss = torch.tensor(0.0, requires_grad=True)
        if self.train_x1_denoising:
            loss_x1, feature_loss_x1, pos_loss_x1, bond_loss_x1 = self.x1_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x1
            
            batch_size_nodes = (~input_dict['x1']['decoder']['virtual_node_mask']).sum().item()
            batch_size_edges = input_dict['x1']['decoder']['bond_edge_x_noise'].shape[0]
            
            self.log('train_loss_x1', loss_x1, batch_size = batch_size_nodes)
            self.log('train_pos_loss_x1', pos_loss_x1, batch_size = batch_size_nodes)
            self.log('train_feature_loss_x1', feature_loss_x1, batch_size = batch_size_nodes)
            self.log('train_bond_loss_x1', bond_loss_x1, batch_size = batch_size_edges)
            
        if self.train_x2_denoising:
            loss_x2 = self.x2_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x2
            
            batch_size_nodes = (~input_dict['x2']['decoder']['virtual_node_mask']).sum().item()
            
            self.log('train_loss_x2', loss_x2, batch_size = batch_size_nodes)
            
        if self.train_x3_denoising:
            loss_x3, feature_loss_x3, pos_loss_x3 = self.x3_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x3
            
            batch_size_nodes = (~input_dict['x3']['decoder']['virtual_node_mask']).sum().item()
            
            self.log('train_loss_x3', loss_x3, batch_size = batch_size_nodes)
            self.log('train_pos_loss_x3', pos_loss_x3, batch_size = batch_size_nodes)
            self.log('train_feature_loss_x3', feature_loss_x3, batch_size = batch_size_nodes)
        
        if self.train_x4_denoising:
            loss_x4, feature_loss_x4, pos_loss_x4, direction_loss_x4 = self.x4_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x4
            
            batch_size_nodes = (~input_dict['x4']['decoder']['virtual_node_mask']).sum().item()
            
            self.log('train_loss_x4', loss_x4, batch_size = batch_size_nodes)
            self.log('train_pos_loss_x4', pos_loss_x4, batch_size = batch_size_nodes)
            self.log('train_direction_loss_x4', direction_loss_x4, batch_size = batch_size_nodes)
            self.log('train_feature_loss_x4', feature_loss_x4, batch_size = batch_size_nodes)
        
        self.log('train_loss', loss, batch_size = batch_size)
        return loss
    

    def x1_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x1']['decoder']['virtual_node_mask']
        pos_loss = torch.mean(
                (input_dict['x1']['decoder']['pos_noise'] - output_dict['x1']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        # ========================= MODIFICATION START: Discrete Feature Loss =========================
        # --- 原子类型损失 (离散，使用交叉熵) ---
        # 模型的输出是 logits: [num_atoms, num_atom_types]
        pred_atom_logits = output_dict['x1']['decoder']['denoiser']['x_out']
        
        # 真实的标签需要是类别索引 (LongTensor)
        # 我们假设 true_atom_types_t0 是 one-hot 编码，所以用 argmax 获取索引
        true_atom_labels = torch.argmax(input_dict['x1']['decoder']['true_atom_types_t0'], dim=1)
        
        # 计算损失时只考虑非虚拟节点
        feature_loss = F.cross_entropy(pred_atom_logits[mask], true_atom_labels[mask])

        # --- 键类型损失 (离散，使用交叉熵) ---
        bond_loss = torch.zeros_like(feature_loss)
        if self.model.x1_bond_diffusion:
            pred_bond_logits = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out']
            true_bond_labels = torch.argmax(input_dict['x1']['decoder']['true_bond_types_t0'], dim=1)
            
            # 这里不需要像之前那样对 real/non-bond 分开加权了，
            # 交叉熵本身就可以处理多分类问题。
            bond_loss = F.cross_entropy(pred_bond_logits, true_bond_labels)
        # ========================= MODIFICATION END ================================================
            
        loss = pos_loss + feature_loss + bond_loss
        
        return loss, feature_loss, pos_loss, bond_loss

    
    def x2_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x2']['decoder']['virtual_node_mask']
        pos_loss = torch.mean(
                (input_dict['x2']['decoder']['pos_noise'] - output_dict['x2']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        loss = pos_loss
        
        return loss
    
    
    def x3_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x3']['decoder']['virtual_node_mask']
        feature_loss = torch.mean(
            (input_dict['x3']['decoder']['x_noise'] - output_dict['x3']['decoder']['denoiser']['x_out'].squeeze())[mask] ** 2.0
        )
        
        pos_loss = torch.mean(
            (input_dict['x3']['decoder']['pos_noise'] - output_dict['x3']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )

# In shepherd/lightning_module.py, within LightningModule

    import torch.nn.functional as F

    def x1_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x1']['decoder']['virtual_node_mask']
        
        # --- 坐标损失 (连续，保持不变) ---
        pos_loss = torch.mean(
                (input_dict['x1']['decoder']['pos_noise'] - output_dict['x1']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        # ========================= MODIFICATION START: Discrete Feature Loss =========================
        # --- 原子类型损失 (离散，使用交叉熵) ---
        # 模型的输出是 logits: [num_atoms, num_atom_types]
        pred_atom_logits = output_dict['x1']['decoder']['denoiser']['x_out']
        
        # 真实的标签需要是类别索引 (LongTensor)
        # 我们假设 true_atom_types_t0 是 one-hot 编码，所以用 argmax 获取索引
        true_atom_labels = torch.argmax(input_dict['x1']['decoder']['true_atom_types_t0'], dim=1)
        
        # 计算损失时只考虑非虚拟节点
        feature_loss = F.cross_entropy(pred_atom_logits[mask], true_atom_labels[mask])

        # --- 键类型损失 (离散，使用交叉熵) ---
        bond_loss = torch.zeros_like(feature_loss)
        if self.model.x1_bond_diffusion:
            pred_bond_logits = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out']
            true_bond_labels = torch.argmax(input_dict['x1']['decoder']['true_bond_types_t0'], dim=1)
            
            # 这里不需要像之前那样对 real/non-bond 分开加权了，
            # 交叉熵本身就可以处理多分类问题。
            bond_loss = F.cross_entropy(pred_bond_logits, true_bond_labels)
        # ========================= MODIFICATION END ================================================

        loss = pos_loss + feature_loss + bond_loss
        
        return loss, feature_loss, pos_loss, bond_loss

    
    def x2_denoising_loss(self, input_dict, output_dict):
        # 这个损失函数只处理坐标，所以保持不变
        mask = ~input_dict['x2']['decoder']['virtual_node_mask']
        pos_loss = torch.mean(
                (input_dict['x2']['decoder']['pos_noise'] - output_dict['x2']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        loss = pos_loss
        return loss
    
    
    def x3_denoising_loss(self, input_dict, output_dict):
        # 这个损失函数处理连续特征，所以保持不变
        mask = ~input_dict['x3']['decoder']['virtual_node_mask']
        feature_loss = torch.mean(
            (input_dict['x3']['decoder']['x_noise'] - output_dict['x3']['decoder']['denoiser']['x_out'].squeeze())[mask] ** 2.0
        )
        pos_loss = torch.mean(
            (input_dict['x3']['decoder']['pos_noise'] - output_dict['x3']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        loss = feature_loss + pos_loss
        return loss, feature_loss, pos_loss
    
    
    def x4_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x4']['decoder']['virtual_node_mask']
        if sum(mask) == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        # --- 坐标和方向损失 (连续，保持不变) ---
        pos_loss = torch.mean(
                (input_dict['x4']['decoder']['pos_noise'] - output_dict['x4']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        direction_loss = torch.mean(
                (input_dict['x4']['decoder']['direction_noise'] - output_dict['x4']['decoder']['denoiser']['direction_out'])[mask] ** 2.0
        )
        
        # ========================= MODIFICATION START: Discrete Feature Loss =========================
        # --- 药效团类型损失 (离散，使用交叉熵) ---
        pred_pharm_logits = output_dict['x4']['decoder']['denoiser']['x_out']
        true_pharm_labels = torch.argmax(input_dict['x4']['decoder']['true_pharm_types_t0'], dim=1)
        
        feature_loss = F.cross_entropy(pred_pharm_logits[mask], true_pharm_labels[mask])
        # ========================= MODIFICATION END ================================================
        
        loss = feature_loss + pos_loss
        
        return loss, feature_loss, pos_loss
    
    
    def x4_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x4']['decoder']['virtual_node_mask']
        if sum(mask) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        feature_loss = torch.mean(
            (input_dict['x4']['decoder']['x_noise'] - output_dict['x4']['decoder']['denoiser']['x_out'].squeeze())[mask] ** 2.0
        )
        
        pos_loss = torch.mean(
                (input_dict['x4']['decoder']['pos_noise'] - output_dict['x4']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        direction_loss = torch.mean(
                (input_dict['x4']['decoder']['direction_noise'] - output_dict['x4']['decoder']['denoiser']['direction_out'])[mask] ** 2.0
        )
        
        loss = feature_loss + pos_loss + direction_loss
        
        return loss, feature_loss, pos_loss, direction_loss
        