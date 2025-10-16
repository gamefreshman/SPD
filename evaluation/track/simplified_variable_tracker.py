#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的变量追踪器

本脚本提供了一个精简而高效的变量追踪工具，其核心使命是在模型推理的复杂过程中，
对指定的关键变量进行实时监控。它特别为深度学习生成模型设计，能够胜任以下任务：

1.  **中间变量监控**:
    在模型的多步生成过程中，持续追踪如原子特征（x1_x_out）、化学键特征（x1_bond_edge_x_out）
    以及药效团位置（x4_pos_out）等关键中间变量的动态变化。

2.  **数值稳定性诊断**:
    自动检测追踪数据中是否出现非数值（NaN）或无穷大（inf）等异常值，并立即发出警告。
    这对于调试模型的数值稳定性和训练过程至关重要。

3.  **数据持久化**:
    将整个追踪会话的所有数据，包括元信息、变量值、警告记录等，系统地保存为结构化的
    JSON 文件。这为后续的性能分析、结果复现和问题排查提供了坚实的数据基础。

"""

# ==================================================================================================
# 依赖导入
# ==================================================================================================
import torch
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional

# ==================================================================================================
# 配置区域
# ==================================================================================================
# --- 日志记录器配置 ---
# 使用 __name__ 作为日志记录器的名称，确保其与模块名精确对应，便于日志管理和过滤。
logger = logging.getLogger(__name__)

# --- 需要追踪的目标变量 --- 
# 此列表定义了在模型推理期间需要被重点监控的关键变量名称。
# 追踪器会主动在模型的输出数据结构中搜索这些变量。
TARGET_VARIABLES = [
    'x1_x_out',           # 原子特征的输出
    'x1_bond_edge_x_out', # 化学键（边）特征的输出
    'x4_pos_out',         # 药效团位置的输出
    'x4_direction_out',   # 药效团方向的输出
    'x4_x_out'            # 药效团特征的输出
]

# ==================================================================================================
# 类定义
# ==================================================================================================
class SimplifiedVariableTracker:
    """
    一个为模型推理过程设计的简化版变量追踪器。

    该类的核心功能是提供一个轻量级的框架，用于实现以下目标：
    1.  **定向追踪**: 根据预设的 `TARGET_VARIABLES` 列表，精确捕获模型在每一步推理中产生的指定变量的数值。
    2.  **异常检测**: 实时扫描捕获到的数据，检测是否存在 NaN 值，并生成详细的警告信息，帮助开发者快速定位数值问题。
    3.  **数据转换**: 将 PyTorch 张量（Tensor）和 NumPy 数组（ndarray）等非原生 Python 类型，安全地转换为列表等 JSON 兼容格式，确保数据的可序列化性和可读性。
    4.  **会话归档**: 将包含元数据、所有追踪步骤的变量值以及警告信息的完整会话，持久化为格式清晰的 JSON 文件，以供深入分析。
    """

    def __init__(self, model, device: torch.device):
        """
        初始化简化变量追踪器实例。

        参数:
            model: 需要被追踪的模型实例。注意：由于 TorchScript 的 `ScriptModule` 类型不支持 `register_forward_hook`，
                   本追踪器目前不通过钩子（hooks）自动绑定到模型层，而是依赖外部调用 `track_inference_step` 方法来手动送入数据。
            device (torch.device): 指定计算设备（例如 'cuda' 或 'cpu'），用于数据处理过程中的设备保持一致。
        """
        self.model = model
        self.device = device
        self.target_variables = TARGET_VARIABLES
        self.tracking_data = {
            'metadata': {},  # 用于存储关于本次追踪会话的元信息，如模型配置、时间戳等
            'variables': {var: [] for var in self.target_variables},  # 分类存储每个目标变量在各个步骤中的数据
            'warnings': [],  # 汇总在整个追踪过程中产生的所有警告信息
            'step_count': 0  # 记录已追踪的总步数
        }
        # 钩子功能当前被禁用，因为它们与 ScriptModule 不兼容。
        self.hooks = []

    def track_inference_step(self, step_number: int, model_output: Any) -> Dict:
        """
        追踪并记录单个推理步骤中产生的模型输出。
        这是与外部代码交互的核心接口，应在模型每执行一步后被调用。

        参数:
            step_number (int): 当前推理步骤的序号（从 0 开始）。
            model_output (Any): 模型在当前步骤的原始输出。这可以是一个复杂的嵌套字典或对象。

        返回:
            Dict: 一个字典，包含了对当前步骤进行追踪和分析后的详细结果，包括找到的变量和产生的警告。
        """
        # 调用内部方法处理当前步骤的数据
        step_result = self._track_variables_in_step(model_output, step_number)

        # 将当前步骤的追踪结果整合到主数据结构 `self.tracking_data` 中
        for var_name in self.target_variables:
            if step_result['variables_found'][var_name]['found']:
                self.tracking_data['variables'][var_name].append({
                    'step': step_number,
                    'data': step_result['variables_found'][var_name]
                })

        # 更新总的追踪步数
        self.tracking_data['step_count'] = max(self.tracking_data['step_count'], step_number + 1)

        return step_result

    def _track_variables_in_step(self, step_data: Dict, step_number: int) -> Dict:
        """
        内部核心方法，负责处理和追踪单个步骤中的所有目标变量。

        参数:
            step_data (Dict): 当前步骤的模型输出数据。
            step_number (int): 当前步骤的序号。

        返回:
            Dict: 一个包含当前步骤详细追踪结果的字典，包括时间戳、找到的变量及其数据、以及该步骤产生的警告。
        """
        step_result = {
            'step': step_number,
            'timestamp': time.time(),
            'variables_found': {},
            'warnings': []
        }

        for var_name in self.target_variables:
            # 提取目标变量的数据
            var_data = self._extract_variable_data(step_data, var_name)
            step_result['variables_found'][var_name] = var_data

            # 检查数据中是否存在 NaN 值，如果存在，则生成并记录警告
            nan_warning = self._check_nan_values(var_data, var_name, step_number)
            if nan_warning:
                step_result['warnings'].append(nan_warning)
                self.tracking_data['warnings'].append(nan_warning) # 同时在会话全局警告列表中记录

        return step_result

    def _extract_variable_data(self, step_data: Dict, var_name: str) -> Dict:
        """
        从给定的步骤数据中，提取、处理并序列化指定变量的数据。

        参数:
            step_data (Dict): 当前步骤的模型输出数据。
            var_name (str): 需要提取的目标变量的名称。

        返回:
            Dict: 一个字典，详细记录了变量的查找结果、序列化后的数据、原始形状和数据类型。
        """
        result = {
            'found': False,  # 标记是否成功找到该变量
            'data': None,    # 存储序列化后的数据（如列表）
            'shape': None,   # 存储变量的原始形状（如 [1, 10, 3]）
            'dtype': None    # 存储变量的原始数据类型（如 'torch.float32'）
        }

        # 在复杂的数据结构中递归搜索目标变量
        var_value = self._search_variable_in_data(step_data, var_name)

        if var_value is not None:
            result['found'] = True
            # 根据变量类型，进行相应的处理和序列化
            if isinstance(var_value, torch.Tensor):
                result['data'] = var_value.detach().cpu().numpy().tolist() # Tensor -> list
                result['shape'] = list(var_value.shape)
                result['dtype'] = str(var_value.dtype)
            elif isinstance(var_value, np.ndarray):
                result['data'] = var_value.tolist() # ndarray -> list
                result['shape'] = list(var_value.shape)
                result['dtype'] = str(var_value.dtype)
            else:
                # 对于标量等其他类型，直接记录
                result['data'] = var_value
                result['shape'] = 'scalar'
                result['dtype'] = type(var_value).__name__

        return result

    def _search_variable_in_data(self, data: Any, var_name: str) -> Any:
        """
        在嵌套的数据结构（如字典、列表、对象）中，通过名称递归地搜索变量。

        参数:
            data (Any): 待搜索的数据结构。
            var_name (str): 要查找的变量名。

        返回:
            Any: 如果找到，返回变量的值；否则返回 None。
        """
        if isinstance(data, dict):
            if var_name in data:
                return data[var_name]
            for key, value in data.items():
                # 模糊匹配：检查变量名是否作为键名的一部分存在
                if var_name in key.lower():
                    return value
                result = self._search_variable_in_data(value, var_name)
                if result is not None:
                    return result
        elif isinstance(data, (list, tuple)):
            for item in data:
                result = self._search_variable_in_data(item, var_name)
                if result is not None:
                    return result
        elif hasattr(data, '__dict__'):
            # 如果是自定义对象，则在其属性字典中搜索
            return self._search_variable_in_data(data.__dict__, var_name)

        return None

    def _check_nan_values(self, var_data: Dict, var_name: str, step_number: int) -> Optional[Dict]:
        """
        检查提取出的变量数据中是否包含 NaN（非数值）值。

        参数:
            var_data (Dict): 包含变量序列化后数据的字典。
            var_name (str): 变量的名称。
            step_number (int): 当前的步骤序号。

        返回:
            Optional[Dict]: 如果检测到 NaN，返回一个包含详细信息的警告字典；否则返回 None。
        """
        if not var_data['found'] or var_data['data'] is None:
            return None

        try:
            data = var_data['data']
            if isinstance(data, list):
                # 定义一个递归辅助函数来深度检查嵌套列表中的 NaN
                def has_nan(item: Any) -> bool:
                    if isinstance(item, list):
                        return any(has_nan(sub_item) for sub_item in item)
                    # 利用 NaN 的特性（NaN != NaN）进行判断
                    return isinstance(item, float) and item != item

                if has_nan(data):
                    logger.warning(f"在步骤 {step_number} 的变量 '{var_name}' 中检测到 NaN 值。")
                    return {
                        'step': step_number,
                        'variable': var_name,
                        'warning_type': 'nan_detected',
                        'message': f'变量 {var_name} 在步骤 {step_number} 中包含 NaN 值。'
                    }
        except Exception as e:
            logger.error(f"在为变量 '{var_name}' 检查 NaN 值时发生错误: {e}")

        return None

    def set_metadata(self, metadata: Dict):
        """
        为本次追踪会话设置或更新元数据。

        参数:
            metadata (Dict): 一个包含元数据的字典，例如模型配置、实验ID、时间戳等。
        """
        self.tracking_data['metadata'].update(metadata)
        logger.info(f"追踪器元数据已更新: {metadata}")

    def get_tracking_summary(self) -> Dict:
        """
        生成并返回整个追踪会话的统计摘要。

        返回:
            Dict: 一个摘要字典，包含了总步数、总警告数以及每个变量的追踪统计信息。
        """
        summary = {
            'total_steps': self.tracking_data['step_count'],
            'total_warnings': len(self.tracking_data['warnings']),
            'variables_stats': {}
        }

        for var_name in self.target_variables:
            var_steps = self.tracking_data['variables'][var_name]
            found_count = sum(1 for step in var_steps if step.get('data', {}).get('found', False))
            summary['variables_stats'][var_name] = {
                'found_count': found_count, # 该变量被成功找到的次数
                'found_rate': found_count / max(1, self.tracking_data['step_count']) # 该变量的发现率
            }

        return summary

    def save_tracking_data(self, filepath: str) -> bool:
        """
        将完整的追踪数据（包括元数据、所有步骤的变量值和警告）保存到指定的 JSON 文件中。

        参数:
            filepath (str): 输出的 JSON 文件的完整路径。

        返回:
            bool: 如果保存成功，返回 True；否则返回 False。
        """
        try:
            # 在保存前，最后一次递归清理数据，确保所有内容都是 JSON 可序列化的
            save_data = self._prepare_data_for_save(self.tracking_data)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.info(f"追踪数据已成功保存至: {filepath}")
            return True
        except Exception as e:
            logger.error(f"无法将追踪数据保存至 {filepath}: {e}")
            return False

    def _prepare_data_for_save(self, data: Any) -> Any:
        """
        递归地准备数据，将其转换为 JSON 可序列化的格式。
        这是一个保障函数，用于处理在 `_extract_variable_data` 中可能遗漏的特殊数据类型。

        参数:
            data (Any): 需要进行转换的数据。

        返回:
            Any: 一个完全 JSON 兼容版本的数据。
        """
        if isinstance(data, dict):
            return {key: self._prepare_data_for_save(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_data_for_save(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._prepare_data_for_save(item) for item in data)
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.bool_, bool)):
            return bool(data)
        elif isinstance(data, (np.integer, int)):
            return int(data)
        elif isinstance(data, (np.floating, float)):
            return float(data)
        elif isinstance(data, (str, type(None))):
            return data
        else:
            # 对于任何其他无法序列化的类型，提供一个回退方案，将其转换为字符串
            try:
                return str(data)
            except Exception:
                return "<无法序列化的对象>"