#!/usr/bin/env python3
"""
测试运行环境脚本
在运行主评估脚本之前验证所有依赖项是否正确安装
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试所有必要的包是否可以导入"""
    print("🔍 检查Python包导入...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('rdkit', 'RDKit'),
        ('tqdm', 'tqdm'),
        ('pickle', 'pickle'),
        ('json', 'json'),
        ('logging', 'logging')
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
    
    # 测试shepherd相关包
    shepherd_packages = [
        ('shepherd.lightning_module', 'Shepherd Lightning Module'),
        ('shepherd.inference', 'Shepherd Inference'),
        ('shepherd.extract', 'Shepherd Extract'),
        ('shepherd_score.evaluations.evaluate', 'Shepherd Score Evaluations'),
        ('shepherd_score.container', 'Shepherd Score Container')
    ]
    
    print("\n🔍 检查Shepherd包导入...")
    for package, name in shepherd_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
    
    return failed_imports

def test_file_paths():
    """测试必要的文件路径是否存在"""
    print("\n🔍 检查文件路径...")
    
    # 从配置文件读取路径
    config_file = Path("config.json")
    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 检查模型检查点
        checkpoint_path = Path(config['model']['checkpoint_path'])
        if checkpoint_path.exists():
            print(f"  ✅ 模型检查点: {checkpoint_path}")
        else:
            print(f"  ❌ 模型检查点不存在: {checkpoint_path}")
        
        # 检查数据文件
        data_path = Path(config['data']['molblocks_path'])
        if data_path.exists():
            print(f"  ✅ 数据文件: {data_path}")
        else:
            print(f"  ❌ 数据文件不存在: {data_path}")
    else:
        print(f"  ❌ 配置文件不存在: {config_file}")
    
    # 检查输出目录
    output_dir = Path("data")
    if output_dir.exists():
        print(f"  ✅ 输出目录: {output_dir}")
    else:
        print(f"  ⚠️ 输出目录不存在，将自动创建: {output_dir}")

def test_cuda():
    """测试CUDA是否可用"""
    print("\n🔍 检查CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"  ✅ CUDA可用")
            print(f"  📊 设备数量: {device_count}")
            print(f"  🖥️ 当前设备: {device_name}")
            
            # 测试GPU内存
            memory_total = torch.cuda.get_device_properties(current_device).total_memory
            memory_total_gb = memory_total / (1024**3)
            print(f"  💾 GPU内存: {memory_total_gb:.1f} GB")
            
            if memory_total_gb < 4:
                print(f"  ⚠️ GPU内存可能不足，推荐至少8GB")
        else:
            print(f"  ❌ CUDA不可用，将使用CPU (速度较慢)")
    except ImportError:
        print(f"  ❌ 无法导入torch检查CUDA")

def main():
    """主测试函数"""
    print("🧪 环境测试开始...")
    print("=" * 60)
    
    # 测试包导入
    failed_imports = test_imports()
    
    # 测试文件路径
    test_file_paths()
    
    # 测试CUDA
    test_cuda()
    
    print("\n" + "=" * 60)
    if failed_imports:
        print(f"❌ 测试完成，但有 {len(failed_imports)} 个包导入失败:")
        for pkg in failed_imports:
            print(f"   - {pkg}")
        print("\n请先安装缺失的依赖包")
        return False
    else:
        print("✅ 所有测试通过! 环境配置正确")
        print("🚀 可以运行主评估脚本了")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
