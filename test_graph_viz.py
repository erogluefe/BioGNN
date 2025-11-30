#!/usr/bin/env python3
"""
Basit graf görselleştirme testi

Bu script, sadece graf yapısını test etmek için minimal bağımlılıklarla çalışır.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Direkt import (dependency zincirini atlayarak)
# biognn/__init__ yüklemeden direkt modülleri import et
import importlib.util

# Base path'i dinamik olarak bul
BASE_DIR = Path(__file__).parent.absolute()

# graph_builder modülünü manuel yükle
spec = importlib.util.spec_from_file_location(
    "graph_builder",
    str(BASE_DIR / "biognn" / "fusion" / "graph_builder.py")
)
graph_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_builder)

ModalityGraphBuilder = graph_builder.ModalityGraphBuilder
AdaptiveEdgeWeighting = graph_builder.AdaptiveEdgeWeighting

# graph_viz modülünü manuel yükle
spec = importlib.util.spec_from_file_location(
    "graph_viz",
    str(BASE_DIR / "biognn" / "visualization" / "graph_viz.py")
)
graph_viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_viz)

GraphVisualizer = graph_viz.GraphVisualizer
plot_graph_structure = graph_viz.plot_graph_structure
plot_edge_strategies_comparison = graph_viz.plot_edge_strategies_comparison
plot_attention_weights = graph_viz.plot_attention_weights

print("="*80)
print("GNN GRAF GÖRSELLEŞTİRME TESTİ")
print("="*80)

# Test 1: Temel graf yapısı
print("\n[TEST 1] Temel Graf Yapısı")
modalities = ['face', 'fingerprint', 'iris', 'voice']
builder = ModalityGraphBuilder(modalities, edge_strategy='fully_connected')

viz = GraphVisualizer(modalities, builder.edge_index)
viz.print_statistics()

fig = viz.plot(layout='circular', title='Test: Fully Connected Graph')
plt.savefig('outputs/test_graph_basic.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ outputs/test_graph_basic.png kaydedildi")

# Test 2: Edge stratejileri karşılaştırması
print("\n[TEST 2] Edge Stratejileri Karşılaştırması")
strategies = ['fully_connected', 'star', 'hierarchical']
for strategy in strategies:
    builder = ModalityGraphBuilder(modalities, edge_strategy=strategy)
    print(f"  {strategy:20s}: {builder.edge_index.shape[1]} kenar")

fig = plot_edge_strategies_comparison(modalities, strategies, builder_class=ModalityGraphBuilder)
plt.savefig('outputs/test_strategies_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ outputs/test_strategies_comparison.png kaydedildi")

# Test 3: Adaptif edge weights
print("\n[TEST 3] Adaptif Edge Weights")
builder = ModalityGraphBuilder(modalities, edge_strategy='fully_connected')

# Dummy modalite özellikleri
feature_dim = 512
node_features = torch.randn(len(modalities), feature_dim)

edge_weighter = AdaptiveEdgeWeighting(
    num_modalities=len(modalities),
    feature_dim=feature_dim,
    hidden_dim=128
)

with torch.no_grad():
    edge_weights = edge_weighter(node_features, builder.edge_index)

print(f"Edge weights stats:")
print(f"  Min: {edge_weights.min().item():.4f}")
print(f"  Max: {edge_weights.max().item():.4f}")
print(f"  Mean: {edge_weights.mean().item():.4f}")

viz = GraphVisualizer(modalities, builder.edge_index, edge_weights)
fig = viz.plot(
    layout='spring',
    show_edge_weights=True,
    title='Test: Adaptive Edge Weights'
)
plt.savefig('outputs/test_adaptive_weights.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ outputs/test_adaptive_weights.png kaydedildi")

# Test 4: Attention weights heatmap
print("\n[TEST 4] Attention Weights Heatmap")
attention_matrix = torch.rand(len(modalities), len(modalities))
attention_matrix = torch.softmax(attention_matrix, dim=1)

fig = plot_attention_weights(
    attention_matrix,
    modalities,
    title='Test: Attention Heatmap'
)
plt.savefig('outputs/test_attention_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ outputs/test_attention_heatmap.png kaydedildi")

print("\n" + "="*80)
print("TÜM TESTLER BAŞARILI!")
print("="*80)
print("\nOluşturulan dosyalar:")
print("  - outputs/test_graph_basic.png")
print("  - outputs/test_strategies_comparison.png")
print("  - outputs/test_adaptive_weights.png")
print("  - outputs/test_attention_heatmap.png")
