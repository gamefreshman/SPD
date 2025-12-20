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

        # DPO 参考模型 初始化
        self.enable_dpo = params['training'].get('enable_dpo', False)
        if self.enable_dpo:
            self.ref_model = Model(params)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # beta_dpo，dpo 的 模型与参考模型的偏离度
        # dpo_max_weight，dpo 的 DPO损失与标准损失的混合比例的最高权重
        self.beta_dpo = params['training'].get('beta_dpo', 0.2)
        self.dpo_ramp_up_epochs = params['training'].get('dpo_ramp_up_epochs', 5)
        self.dpo_max_weight = params['training'].get('dpo_max_weight', 0.5)
        
    def get_dpo_weight(self):
        if not self.enable_dpo:
            return 0.0
        
        epoch = self.current_epoch
        if epoch < self.dpo_ramp_up_epochs:
            return  (epoch + 1) * self.dpo_max_weight / self.dpo_ramp_up_epochs
        else:
            return self.dpo_max_weight

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        覆盖 PyTorch Lightning 的默认加载行为。
        当从一个包含了旧模型（例如，包含 x3_decoder）权重的检查点文件恢复训练时，
        而当前模型配置又不包含这些模块（如此处 explicit_diffusion_variables 不含 'x3'），
        默认的 strict=True 会导致“Unexpected key(s)”错误。

        通过将 strict 设置为 False，我们告诉加载器忽略那些在检查点中存在但在当前模型中不存在的键，
        从而允许我们灵活地从具有不同配置的旧检查点恢复训练，只加载匹配的权重。
        """
        # 调用父类（torch.nn.Module）的 load_state_dict 方法，但强制 strict=False
        # 这样可以加载所有匹配的权重，并静默地忽略不匹配的权重（例如 x3_decoder 的权重）。
        print("正在以非严格模式加载模型状态，将忽略检查点中不匹配的键...")
        return super().load_state_dict(state_dict, strict=False)
    
    
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
            
            # 1. 获取节点存储
            x1_data = data['x1']
            
            # 2. 获取边存储
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
                    
                    'pos_noise': data['x2'].pos_noise, # this is the added (gaussian) noise
                    'virtual_node_mask': data['x2'].virtual_node_mask,  # 新增
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
                    
                    'pos_noise': data['x3'].pos_noise, # this is the added (gaussian) noise
                    'x_noise': data['x3'].x_noise, # this is the added (gaussian) noise
                    'virtual_node_mask': data['x3'].virtual_node_mask,  # 新增
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

        if isinstance(train_batch, dict) and 'batch_type' in train_batch:
            
            batch_type = train_batch['batch_type']
        
        else:
            
            batch_type = 'standard'

        if batch_type == 'standard':

            data = train_batch

            batch_size = data.molecule_id.shape[0]
            
            input_dict = self.get_training_input_dict(data)
            
            output_dict = self.forward_training(input_dict)
            
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

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

        elif batch_type == 'dpo':
            
            batch_winner = train_batch.get('winner')
            batch_loser = train_batch.get('loser')
            shared_noise = train_batch.get('shared_noise', {})
            shared_timestep = train_batch.get('shared_timestep', 0.5)
            
            # DDP安全检查：验证批次数据有效性
            if batch_winner is None or batch_loser is None:
                # 某些rank可能没有数据，返回零损失以保持DDP同步
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 检查batch是否有有效的x1数据
            if not hasattr(batch_winner, 'x1') or batch_winner['x1'] is None:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            input_winner = self.get_training_input_dict(batch_winner)
            output_winner = self.forward_training(input_winner)
            loss_std, _, _, _ = self.x1_denoising_loss(input_winner, output_winner)

            loss_dpo, implicit_acc, model_diff, ref_diff = self.compute_dpo_loss(
                batch_winner, batch_loser, shared_noise, shared_timestep
            )

            dpo_weight = self.get_dpo_weight()

            loss = (1 - dpo_weight) * loss_std + dpo_weight * loss_dpo

            self.log('dpo_weight', dpo_weight)
            self.log('implicit_acc', implicit_acc)
            self.log('model_loss_diff', model_diff)
            self.log('ref_loss_diff', ref_diff)
            self.log('loss_dpo', loss_dpo)
            self.log('loss_std_on_winner', loss_std)
            self.log('train_loss', loss)
            
            return loss

        # DDP模式下的安全回退：如果batch_type未知或批次无效，返回零损失
        # 这避免了在分布式训练中因返回None而导致的RuntimeError
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def x1_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x1']['decoder']['virtual_node_mask']
        pos_loss = torch.mean(
                (input_dict['x1']['decoder']['pos_noise'] - output_dict['x1']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        # --- 原子类型损失 (离散，使用交叉熵) ---
        # 模型的输出是 logits: [num_atoms, num_atom_types]
        # pred_atom_logits 是模型预测的原子类型概率分布(logits形式)
        # shape: [num_atoms, num_atom_types]
        # 对每个原子,输出num_atom_types个数字,经过softmax后表示属于每种原子类型的概率
        # 数值越大,表示模型越倾向于预测该位置是对应的原子类型
        pred_atom_logits = output_dict['x1']['decoder']['denoiser']['x_out']
        # 真实的标签需要是类别索引 (LongTensor)
        # 我们假设 true_atom_types_t0 是 one-hot 编码，所以用 argmax 获取索引
        # 将原子类型的one-hot编码转换为类别索引
        # 例如: [0,1,0,0] -> 1, [0,0,1,0] -> 2
        # dim=1表示在第二个维度上取最大值的索引
        true_atom_labels = torch.argmax(input_dict['x1']['decoder']['true_atom_types_t0'], dim=1)
        # 计算损失时只考虑非虚拟节点
        # 计算原子类型预测的交叉熵损失
        # pred_atom_logits[mask]: 模型预测的原子类型logits, shape为[num_real_atoms, num_atom_types]
        # true_atom_labels[mask]: 真实的原子类型标签(类别索引), shape为[num_real_atoms]
        # 交叉熵将模型输出的logits(每个原子类型的得分)与真实的单一类别标签进行比较，计算预测错误的惩罚值
        feature_loss = F.cross_entropy(pred_atom_logits[mask], true_atom_labels[mask])
        # --- 键类型损失 (离散，使用交叉熵) ---
        bond_loss = torch.zeros_like(feature_loss)
        if self.model.x1_bond_diffusion:
            pred_bond_logits = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out']
            true_bond_labels = torch.argmax(input_dict['x1']['decoder']['true_bond_types_t0'], dim=1)
            
            # 这里不需要像之前那样对 real/non-bond 分开加权了，
            # 交叉熵本身就可以处理多分类问题。
            bond_loss = F.cross_entropy(pred_bond_logits, true_bond_labels)
            
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
        
        # 检查是否有非虚拟节点，避免在空张量上计算损失
        if mask.sum() == 0:
            zero_loss = torch.tensor(0.0, device=self.device)
            return zero_loss, zero_loss, zero_loss

        feature_loss = torch.mean(
            (input_dict['x3']['decoder']['x_noise'] - output_dict['x3']['decoder']['denoiser']['x_out'].squeeze())[mask] ** 2.0
        )
        
        pos_loss = torch.mean(
            (input_dict['x3']['decoder']['pos_noise'] - output_dict['x3']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        # 将两个损失相加
        loss = feature_loss + pos_loss
        
        # 返回总损失和分量
        return loss, feature_loss, pos_loss
    
    def x4_denoising_loss(self, input_dict, output_dict):
        mask = ~input_dict['x4']['decoder']['virtual_node_mask']
        if mask.sum() == 0:
            # 增加对空张量的处理，避免在.item()上调用
            # 并且返回四个零张量以匹配函数签名
            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero_loss, zero_loss.detach(), zero_loss.detach(), zero_loss.detach()
        
        # --- 药效团类型损失 (离散, 使用交叉熵) ---
        pred_pharm_logits = output_dict['x4']['decoder']['denoiser']['x_out']
        true_pharm_labels = torch.argmax(input_dict['x4']['decoder']['true_pharm_types_t0'], dim=1)
        
        # 只在非虚拟节点上计算损失
        feature_loss = F.cross_entropy(pred_pharm_logits[mask], true_pharm_labels[mask])

        # --- 连续特征损失 (MSE) ---
        pos_loss = torch.mean(
            (input_dict['x4']['decoder']['pos_noise'] - output_dict['x4']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        direction_loss = torch.mean(
            (input_dict['x4']['decoder']['direction_noise'] - output_dict['x4']['decoder']['denoiser']['direction_out'])[mask] ** 2.0
        )
        
        loss = feature_loss + pos_loss + direction_loss
        
        return loss, feature_loss, pos_loss, direction_loss
        
    def compute_dpo_loss(self, batch_winner, batch_loser,  shared_noise, shared_timestep):
        
        input_winner = self.get_training_input_dict(batch_winner)
        input_loser = self.get_training_input_dict(batch_loser)

        output_model_winner = self.model.forward(input_winner)[1]
        output_model_loser = self.model.forward(input_loser)[1] 

        with torch.no_grad():
            output_ref_winner = self.ref_model.forward(input_winner)[1]
            output_ref_loser = self.ref_model.forward(input_loser)[1]

        loss_dpo_continuous = 0.0
        loss_dpo_discrete = 0.0

        if self.train_x1_denoising:
        # 计算MSE损失
            mask_w = ~input_winner['x1']['decoder']['virtual_node_mask']
            mask_l = ~input_loser['x1']['decoder']['virtual_node_mask']
            
            # Model losses
            noise_pred_w = output_model_winner['x1']['decoder']['denoiser']['pos_out']
            noise_pred_l = output_model_loser['x1']['decoder']['denoiser']['pos_out']
            noise_true_w = input_winner['x1']['decoder']['pos_noise']
            noise_true_l = input_loser['x1']['decoder']['pos_noise']
            
            model_loss_w = torch.mean((noise_pred_w[mask_w] - noise_true_w[mask_w]) ** 2)
            model_loss_l = torch.mean((noise_pred_l[mask_l] - noise_true_l[mask_l]) ** 2)
            
            # Ref losses
            noise_ref_w = output_ref_winner['x1']['decoder']['denoiser']['pos_out']
            noise_ref_l = output_ref_loser['x1']['decoder']['denoiser']['pos_out']
            
            ref_loss_w = torch.mean((noise_ref_w[mask_w] - noise_true_w[mask_w]) ** 2)
            ref_loss_l = torch.mean((noise_ref_l[mask_l] - noise_true_l[mask_l]) ** 2)
            
            # DPO公式
            model_diff = model_loss_w - model_loss_l
            ref_diff = ref_loss_w - ref_loss_l
            inside_term = -0.5 * self.beta_dpo * (model_diff - ref_diff)
            loss_dpo_continuous = -torch.log(torch.sigmoid(inside_term) + 1e-8)

        if self.train_x1_denoising:
            mask_w = ~input_winner['x1']['decoder']['virtual_node_mask']
            mask_l = ~input_loser['x1']['decoder']['virtual_node_mask']
            
            # Model losses - atom types
            logits_w = output_model_winner['x1']['decoder']['denoiser']['x_out']
            logits_l = output_model_loser['x1']['decoder']['denoiser']['x_out']
            true_labels_w = torch.argmax(input_winner['x1']['decoder']['true_atom_types_t0'], dim=1)
            true_labels_l = torch.argmax(input_loser['x1']['decoder']['true_atom_types_t0'], dim=1)
            
            model_loss_w = F.cross_entropy(logits_w[mask_w], true_labels_w[mask_w])
            model_loss_l = F.cross_entropy(logits_l[mask_l], true_labels_l[mask_l])
            
            # Ref losses
            logits_ref_w = output_ref_winner['x1']['decoder']['denoiser']['x_out']
            logits_ref_l = output_ref_loser['x1']['decoder']['denoiser']['x_out']
            
            ref_loss_w = F.cross_entropy(logits_ref_w[mask_w], true_labels_w[mask_w])
            ref_loss_l = F.cross_entropy(logits_ref_l[mask_l], true_labels_l[mask_l])
            
            # DPO公式
            model_diff = model_loss_w - model_loss_l
            ref_diff = ref_loss_w - ref_loss_l
            inside_term = -0.5 * self.beta_dpo * (model_diff - ref_diff)
            loss_dpo_discrete = -torch.log(torch.sigmoid(inside_term) + 1e-8)

        implicit_acc =  (model_diff < 0).float().mean()

        loss_dpo_total = loss_dpo_continuous + loss_dpo_discrete

        return loss_dpo_total, implicit_acc, model_diff, ref_diff