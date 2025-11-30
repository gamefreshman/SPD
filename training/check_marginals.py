#!/usr/bin/env python3
"""检查边际分布差异"""
import torch

# 加载两个数据集的边际分布
nps_atom = torch.load('cached_marginals/NPs_atom_marginals.pt', weights_only=True)
moses_atom = torch.load('cached_marginals/MOSES_aq_atom_marginals.pt', weights_only=True)

print("=" * 60)
print("📊 NPs 边际分布 (3个天然产物分子):")
print("=" * 60)
print(nps_atom)
print(f"\n非零元素索引: {(nps_atom > 0).nonzero(as_tuple=True)[0].tolist()}")
print(f"非零元素数量: {(nps_atom > 0).sum().item()}")

print("\n" + "=" * 60)
print("📊 MOSES-AQ 边际分布 (预训练数据集):")
print("=" * 60)
print(moses_atom)
print(f"\n非零元素索引: {(moses_atom > 0).nonzero(as_tuple=True)[0].tolist()}")
print(f"非零元素数量: {(moses_atom > 0).sum().item()}")

print("\n" + "=" * 60)
print("🔍 关键差异分析:")
print("=" * 60)

atom_types = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']

print("\n索引 -> 原子类型 -> NPs概率 vs MOSES-AQ概率")
print("-" * 60)
for i, atom_name in enumerate(atom_types):
    nps_prob = nps_atom[i].item() if i < len(nps_atom) else 0
    moses_prob = moses_atom[i].item() if i < len(moses_atom) else 0
    symbol = "✓" if nps_prob > 0 else "✗"
    print(f"{symbol} [{i:2d}] {str(atom_name):4s} -> NPs: {nps_prob:8.6f}  |  MOSES: {moses_prob:8.6f}")
