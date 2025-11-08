#!/usr/bin/env python3
"""
重启脚本：停止当前评估进程并启动并行版本
"""

import os
import subprocess
import time
import signal
from pathlib import Path

def kill_existing_processes():
    """终止现有的评估进程"""
    print("🔍 查找现有的分子评估进程...")
    
    try:
        # 查找相关进程
        result = subprocess.run(['pgrep', '-f', 'molecular_evaluation'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"发现 {len(pids)} 个进程: {pids}")
            
            # 优雅终止
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"发送SIGTERM信号到进程 {pid}")
                except ProcessLookupError:
                    print(f"进程 {pid} 已经不存在")
                except Exception as e:
                    print(f"无法终止进程 {pid}: {e}")
            
            # 等待进程退出
            print("等待进程优雅退出...")
            time.sleep(5)
            
            # 检查是否还有进程存在
            result = subprocess.run(['pgrep', '-f', 'molecular_evaluation'], 
                                  capture_output=True, text=True)
            
            if result.stdout.strip():
                remaining_pids = result.stdout.strip().split('\n')
                print(f"强制终止剩余进程: {remaining_pids}")
                
                # 强制终止
                for pid in remaining_pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"强制终止进程 {pid}")
                    except ProcessLookupError:
                        print(f"进程 {pid} 已经不存在")
                    except Exception as e:
                        print(f"无法强制终止进程 {pid}: {e}")
            
            print("✅ 所有相关进程已终止")
        else:
            print("✅ 没有发现运行中的评估进程")
            
    except FileNotFoundError:
        print("⚠️ pgrep命令不可用，尝试其他方法...")
        # 使用ps命令
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'molecular_evaluation' in line and 'python' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                            print(f"强制终止进程 {pid}")
                        except Exception as e:
                            print(f"无法终止进程 {pid}: {e}")
        except Exception as e:
            print(f"进程终止失败: {e}")

def check_gpu_status():
    """检查GPU状态"""
    print("\n🖥️ 检查GPU状态...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_id, name, mem_used, mem_total, util = parts
                    print(f"  GPU {gpu_id}: {name}")
                    print(f"    内存: {mem_used}MB / {mem_total}MB")
                    print(f"    利用率: {util}%")
        else:
            print("⚠️ 无法获取GPU信息")
    except FileNotFoundError:
        print("⚠️ nvidia-smi不可用")

def start_parallel_evaluation():
    """启动并行评估"""
    print("\n🚀 启动并行评估...")
    
    # 检查当前目录
    if not Path('molecular_evaluation.py').exists():
        print("❌ 当前目录中找不到molecular_evaluation.py")
        print("请确保在 /home1/zhh/workspace/SPD/evaluation/core/ 目录下运行此脚本")
        return False
    
    try:
        # 使用nohup后台运行
        cmd = ['nohup', 'python', 'run_evaluation.py', '--skip-test']
        
        with open('parallel_evaluation.log', 'w') as log_file:
            process = subprocess.Popen(cmd, 
                                     stdout=log_file, 
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True)
        
        print(f"✅ 并行评估已启动，进程ID: {process.pid}")
        print("📋 监控命令:")
        print("  tail -f parallel_evaluation.log     # 查看实时日志")
        print("  nvidia-smi                          # 查看GPU使用情况")
        print(f"  kill {process.pid}                  # 停止评估 (如需要)")
        
        return True
        
    except Exception as e:
        print(f"❌ 启动并行评估失败: {e}")
        return False

def main():
    print("🔄 重启分子评估 - 启用并行处理")
    print("=" * 60)
    
    # 1. 终止现有进程
    kill_existing_processes()
    
    # 2. 检查GPU状态
    check_gpu_status()
    
    # 3. 启动并行评估
    if start_parallel_evaluation():
        print("\n🎉 并行评估启动成功!")
        print("💡 预期性能提升:")
        print("  - 采样速度: 提升 2-3倍 (多GPU并行)")
        print("  - GPU利用率: 显著提升")
        print("  - 总时间: 预计减少到 1-2小时")
    else:
        print("\n❌ 并行评估启动失败")

if __name__ == "__main__":
    main()
