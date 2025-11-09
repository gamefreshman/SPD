#!/usr/bin/env python3

import torch
import os
from pathlib import Path

def load_and_print_marginals(cache_dir="cached_marginals"):
    """加载并打印缓存的边际分布数据"""
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"缓存目录 {cache_dir} 不存在")
        return
    
    # 获取所有 .pt 文件
    pt_files = list(cache_path.glob("*.pt"))
    
    if not pt_files:
        print(f"在 {cache_dir} 目录中未找到 .pt 文件")
        return
    
    print(f"=== 缓存边际分布数据 ({cache_dir}) ===\n")
    
    for pt_file in sorted(pt_files):
        print(f"📁 文件: {pt_file.name}")
        print(f"   大小: {pt_file.stat().st_size} bytes")
        
        try:
            # 加载PyTorch tensor
            data = torch.load(pt_file, map_location='cpu')
            
            print(f"   类型: {type(data)}")
            
            if isinstance(data, dict):
                print(f"   字典键值: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: shape={value.shape}, dtype={value.dtype}")
                        print(f"     {key}: min={value.min().item():.4f}, max={value.max().item():.4f}, sum={value.sum().item():.4f}")
                        if value.numel() <= 20:  # 如果元素少于20个，打印全部
                            print(f"     {key}: values={value.tolist()}")
                        else:
                            print(f"     {key}: first 10 values={value.flatten()[:10].tolist()}")
                    else:
                        print(f"     {key}: {value}")
            elif isinstance(data, torch.Tensor):
                print(f"   张量shape: {data.shape}")
                print(f"   数据类型: {data.dtype}")
                print(f"   统计信息: min={data.min().item():.4f}, max={data.max().item():.4f}, sum={data.sum().item():.4f}")
                if data.numel() <= 20:
                    print(f"   数值: {data.tolist()}")
                else:
                    print(f"   前10个数值: {data.flatten()[:10].tolist()}")
            else:
                print(f"   数据: {data}")
                
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    # 切换到SPD项目目录下的training文件夹
    os.chdir("/home1/zhh/workspace/SPD/training")
    load_and_print_marginals("cached_marginals")
