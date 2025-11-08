#!/usr/bin/env python3
"""
运行分子评估的入口脚本
"""

import sys
import os
from pathlib import Path
import argparse

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root))

def run_environment_test():
    """运行环境测试"""
    try:
        from test_environment import main as test_main
        return test_main()
    except ImportError:
        print("❌ 无法导入环境测试脚本")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分子生成与评估")
    parser.add_argument("--skip-test", action="store_true", help="跳过环境测试直接运行")
    args = parser.parse_args()
    
    # 如果没有跳过测试，先进行环境检查
    if not args.skip_test:
        print("🧪 首先进行环境检查...")
        if not run_environment_test():
            print("\n❌ 环境检查失败，请修复问题后重试")
            print("提示: 使用 --skip-test 参数可跳过环境检查")
            sys.exit(1)
        print("\n" + "="*60)
    
    print("🚀 开始分子生成与评估...")
    print("=" * 60)
    # 读取配置显示实际采样数
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        samples_per_mol = config.get('evaluation', {}).get('samples_per_molecule', 'unknown')
    except:
        samples_per_mol = 'unknown'
    
    print("评估配置:")
    print("  - 目标分子数: 3个天然产物")
    print(f"  - 每个分子采样数: {samples_per_mol}个")
    print("  - 评估流程: create_rdkit_molecule -> ConfEval -> ConditionalEval")
    print("  - 输出目录: evaluation/core/data/")
    print("=" * 60)
    
    try:
        from molecular_evaluation import main
        main()
        print("\n🎉 评估完成! 结果已保存到以下文件:")
        data_dir = Path("data")
        if data_dir.exists():
            for json_file in data_dir.glob("*.json"):
                print(f"  - {json_file}")
    except KeyboardInterrupt:
        print("\n⚠️ 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        raise
