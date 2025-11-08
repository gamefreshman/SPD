#!/usr/bin/env python3
"""
清理用户的Python进程
"""

import subprocess
import time
import os
import signal

def get_user_python_processes(username="zhh"):
    """获取指定用户的Python进程"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = []
        
        for line in result.stdout.split('\n'):
            if username in line and 'python' in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    command = ' '.join(parts[10:])
                    processes.append((pid, command))
        
        return processes
    except Exception as e:
        print(f"获取进程列表失败: {e}")
        return []

def kill_processes(processes, force=False):
    """终止进程列表"""
    if not processes:
        print("✅ 没有发现Python进程")
        return
    
    print(f"发现 {len(processes)} 个Python进程:")
    for pid, command in processes:
        print(f"  PID {pid}: {command[:80]}...")
    
    signal_type = signal.SIGKILL if force else signal.SIGTERM
    signal_name = "SIGKILL" if force else "SIGTERM"
    
    print(f"\n🛑 发送 {signal_name} 信号...")
    
    for pid, command in processes:
        try:
            os.kill(int(pid), signal_type)
            print(f"  ✅ 终止进程 {pid}")
        except ProcessLookupError:
            print(f"  ⚠️ 进程 {pid} 已经不存在")
        except PermissionError:
            print(f"  ❌ 没有权限终止进程 {pid}")
        except Exception as e:
            print(f"  ❌ 终止进程 {pid} 失败: {e}")

def main():
    print("🔍 Python进程清理工具")
    print("=" * 40)
    
    # 1. 获取当前用户的Python进程
    processes = get_user_python_processes()
    
    if not processes:
        print("✅ 没有发现需要清理的Python进程")
        return
    
    # 2. 优雅终止
    print("第1步: 优雅终止...")
    kill_processes(processes, force=False)
    
    # 3. 等待进程退出
    print("\n⏳ 等待5秒让进程退出...")
    time.sleep(5)
    
    # 4. 检查剩余进程
    remaining = get_user_python_processes()
    
    if remaining:
        print(f"\n⚠️ 仍有 {len(remaining)} 个进程存在，强制终止...")
        kill_processes(remaining, force=True)
        
        # 再等待2秒
        time.sleep(2)
        final_check = get_user_python_processes()
        
        if final_check:
            print(f"\n❌ 仍有 {len(final_check)} 个进程无法终止:")
            for pid, command in final_check:
                print(f"  PID {pid}: {command[:60]}...")
        else:
            print("\n🎉 所有Python进程已成功清理!")
    else:
        print("\n🎉 所有Python进程已成功清理!")

if __name__ == "__main__":
    main()
