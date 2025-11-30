#!/usr/bin/env python3
"""
GNN Graph Structure Visualization Example

Bu örnek, BioGNN'in multimodal füzyon için kullandığı graf yapılarını
görselleştirir. Farklı edge stratejilerini, adaptif edge weights'leri
ve attention mekanizmalarını gösterir.

Kullanım:
    python examples/visualize_graph_structure.py
    python examples/visualize_graph_structure.py --strategy fully_connected
    python examples/visualize_graph_structure.py --output ./outputs
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from biognn.fusion.graph_builder import ModalityGraphBuilder, AdaptiveEdgeWeighting
from biognn.visualization import (
    GraphVisualizer,
    plot_graph_structure,
    plot_edge_strategies_comparison,
    plot_attention_weights
)


def example_1_basic_graph_structure():
    """
    Örnek 1: Temel Graf Yapısı

    4 modalite (face, fingerprint, iris, voice) ile basit bir graf
    yapısını görselleştirir.
    """
    print("\n" + "="*80)
    print("ÖRNEK 1: TEMEL GRAF YAPISI")
    print("="*80)

    modalities = ['face', 'fingerprint', 'iris', 'voice']

    # Fully connected graph builder
    builder = ModalityGraphBuilder(
        modalities=modalities,
        edge_strategy='fully_connected'
    )

    print(f"\nModaliteler: {modalities}")
    print(f"Edge Strategy: fully_connected")
    print(f"Toplam düğüm sayısı: {len(modalities)}")
    print(f"Toplam kenar sayısı: {builder.edge_index.shape[1]}")

    # Görselleştir
    viz = GraphVisualizer(
        modalities=modalities,
        edge_index=builder.edge_index
    )

    # İstatistikleri yazdır
    viz.print_statistics()

    # Graf çiz
    fig = viz.plot(
        layout='circular',
        title='Fully Connected Multimodal Graph',
        save_path='outputs/graph_basic.png'
    )

    plt.show()
    print("✓ Graf kaydedildi: outputs/graph_basic.png")


def example_2_edge_strategies_comparison():
    """
    Örnek 2: Edge Stratejileri Karşılaştırması

    Üç farklı edge stratejisini yan yana gösterir:
    - Fully Connected: Tüm modaliteler birbirine bağlı
    - Star: Merkezi bir hub'a bağlı (genelde face)
    - Hierarchical: Hiyerarşik zincir yapısı
    """
    print("\n" + "="*80)
    print("ÖRNEK 2: EDGE STRATEJİLERİ KARŞILAŞTIRMASI")
    print("="*80)

    modalities = ['face', 'fingerprint', 'iris', 'voice']
    strategies = ['fully_connected', 'star', 'hierarchical']

    print(f"\nKarşılaştırılan stratejiler:")
    for strategy in strategies:
        builder = ModalityGraphBuilder(modalities, edge_strategy=strategy)
        print(f"  - {strategy:20s}: {builder.edge_index.shape[1]} kenar")

    # Karşılaştırmalı görselleştirme
    fig = plot_edge_strategies_comparison(
        modalities=modalities,
        edge_strategies=strategies,
        save_path='outputs/graph_strategies_comparison.png'
    )

    plt.show()
    print("\n✓ Karşılaştırma kaydedildi: outputs/graph_strategies_comparison.png")


def example_3_adaptive_edge_weights():
    """
    Örnek 3: Adaptif Edge Weights

    Modalite özelliklere göre dinamik olarak öğrenilen edge weights'leri
    görselleştirir. Edge kalınlığı ve rengi, weight değerini gösterir.
    """
    print("\n" + "="*80)
    print("ÖRNEK 3: ADAPTİF EDGE WEIGHTS")
    print("="*80)

    modalities = ['face', 'fingerprint', 'iris', 'voice']

    # Graf builder
    builder = ModalityGraphBuilder(
        modalities=modalities,
        edge_strategy='fully_connected',
        learnable_edges=True
    )

    # Simüle edilmiş modalite özellikleri
    # Gerçek senaryoda bunlar feature extractor'lardan gelir
    batch_size = 8
    feature_dim = 512

    node_features = torch.randn(len(modalities), feature_dim)

    # Adaptif edge weighting modülü
    edge_weighter = AdaptiveEdgeWeighting(
        num_modalities=len(modalities),
        feature_dim=feature_dim,
        hidden_dim=128
    )

    # Edge weights hesapla
    with torch.no_grad():
        edge_weights = edge_weighter(node_features, builder.edge_index)

    print(f"\nModaliteler: {modalities}")
    print(f"Edge weights istatistikleri:")
    print(f"  Min: {edge_weights.min().item():.4f}")
    print(f"  Max: {edge_weights.max().item():.4f}")
    print(f"  Mean: {edge_weights.mean().item():.4f}")
    print(f"  Std: {edge_weights.std().item():.4f}")

    # Edge weights ile görselleştir
    viz = GraphVisualizer(
        modalities=modalities,
        edge_index=builder.edge_index,
        edge_weights=edge_weights
    )

    fig = viz.plot(
        layout='spring',
        show_edge_weights=True,
        show_edge_labels=True,
        title='Adaptive Edge Weights (Learned)',
        save_path='outputs/graph_adaptive_weights.png'
    )

    plt.show()
    print("\n✓ Adaptif edge weights grafiği kaydedildi: outputs/graph_adaptive_weights.png")


def example_4_attention_weights_heatmap():
    """
    Örnek 4: Attention Weights Heatmap

    GAT (Graph Attention Network) tarafından öğrenilen attention
    weights'lerini heatmap olarak görselleştirir. Bu, hangi modalitelerin
    birbirine daha çok dikkat ettiğini gösterir.
    """
    print("\n" + "="*80)
    print("ÖRNEK 4: ATTENTION WEIGHTS HEATMAP")
    print("="*80)

    modalities = ['face', 'fingerprint', 'iris', 'voice']

    # Simüle edilmiş attention weights
    # Gerçek senaryoda bunlar GAT modelinden gelir
    # Face modalitesi genelde en güvenilir olduğu için yüksek attention alır
    attention_matrix = torch.tensor([
        [0.45, 0.25, 0.15, 0.15],  # Face -> all
        [0.35, 0.40, 0.15, 0.10],  # Fingerprint -> all
        [0.30, 0.20, 0.35, 0.15],  # Iris -> all
        [0.25, 0.15, 0.20, 0.40]   # Voice -> all
    ])

    # Normalize (her satır toplam 1 olmalı)
    attention_matrix = torch.softmax(attention_matrix, dim=1)

    print(f"\nModaliteler: {modalities}")
    print(f"Attention matrix shape: {attention_matrix.shape}")
    print(f"\nAttention analizi:")

    for i, src_mod in enumerate(modalities):
        max_attn_idx = attention_matrix[i].argmax().item()
        max_attn_val = attention_matrix[i].max().item()
        target_mod = modalities[max_attn_idx]

        if src_mod == target_mod:
            print(f"  {src_mod:12s} -> en çok kendine dikkat ediyor ({max_attn_val:.3f})")
        else:
            print(f"  {src_mod:12s} -> en çok {target_mod}'e dikkat ediyor ({max_attn_val:.3f})")

    # Heatmap görselleştir
    fig = plot_attention_weights(
        attention_matrix=attention_matrix,
        modalities=modalities,
        title='GAT Attention Weights (Inter-Modality)',
        save_path='outputs/attention_heatmap.png'
    )

    plt.show()
    print("\n✓ Attention heatmap kaydedildi: outputs/attention_heatmap.png")


def example_5_quality_aware_graph():
    """
    Örnek 5: Kalite Skorlarına Göre Ağırlıklandırılmış Graf

    Biyometrik kalite skorlarına göre edge weights'lerin nasıl
    değiştiğini gösterir. Düşük kaliteli modaliteler daha düşük
    bağlantı ağırlıklarına sahip olur.
    """
    print("\n" + "="*80)
    print("ÖRNEK 5: KALİTE SKORLARINA GÖRE AĞIRLIKLANDIRILMIŞ GRAF")
    print("="*80)

    modalities = ['face', 'fingerprint', 'iris', 'voice']

    # Simüle edilmiş kalite skorları (0-1 arası)
    quality_scores = {
        'face': 0.95,        # Yüksek kalite
        'fingerprint': 0.85,  # İyi kalite
        'iris': 0.60,        # Orta kalite (kötü aydınlatma)
        'voice': 0.40        # Düşük kalite (gürültülü ortam)
    }

    print(f"\nKalite Skorları:")
    for mod, score in quality_scores.items():
        quality_label = "Yüksek" if score > 0.8 else "İyi" if score > 0.6 else "Orta" if score > 0.4 else "Düşük"
        print(f"  {mod:12s}: {score:.2f} ({quality_label})")

    # Graf builder
    builder = ModalityGraphBuilder(
        modalities=modalities,
        edge_strategy='fully_connected',
        use_quality_scores=True
    )

    # Kalite skorlarına göre edge weights hesapla
    # Yüksek kaliteli modaliteler arasındaki bağlantılar daha güçlü
    edge_weights = []
    for i in range(builder.edge_index.shape[1]):
        src_idx = builder.edge_index[0, i].item()
        dst_idx = builder.edge_index[1, i].item()

        src_mod = modalities[src_idx]
        dst_mod = modalities[dst_idx]

        # Edge weight = iki modalite kalitesinin ortalaması
        weight = (quality_scores[src_mod] + quality_scores[dst_mod]) / 2
        edge_weights.append(weight)

    edge_weights = torch.tensor(edge_weights)

    print(f"\nEdge weights istatistikleri:")
    print(f"  Min: {edge_weights.min().item():.4f} (en düşük kaliteli bağlantı)")
    print(f"  Max: {edge_weights.max().item():.4f} (en yüksek kaliteli bağlantı)")
    print(f"  Mean: {edge_weights.mean().item():.4f}")

    # Görselleştir
    viz = GraphVisualizer(
        modalities=modalities,
        edge_index=builder.edge_index,
        edge_weights=edge_weights,
        style='presentation'
    )

    fig = viz.plot(
        layout='circular',
        show_edge_weights=True,
        title='Quality-Aware Graph (Edge thickness = Quality)',
        save_path='outputs/graph_quality_aware.png'
    )

    plt.show()
    print("\n✓ Kalite skorları grafiği kaydedildi: outputs/graph_quality_aware.png")


def example_6_real_world_scenario():
    """
    Örnek 6: Gerçek Dünya Senaryosu

    Bir havaalanı güvenlik sistemi senaryosu:
    - 5 modalite (face, fingerprint, iris, voice, gait)
    - Bazı modaliteler mevcut değil (eksik veri)
    - Kalite skorları değişken
    """
    print("\n" + "="*80)
    print("ÖRNEK 6: GERÇEK DÜNYA SENARYOSU - HAVAAALANI GÜVENLİK SİSTEMİ")
    print("="*80)

    modalities = ['face', 'fingerprint', 'iris', 'voice', 'gait']

    print("\nSenaryo:")
    print("  Bir yolcu havaalanı güvenlik kontrolünden geçiyor.")
    print("  Bazı modaliteler eksik veya düşük kaliteli olabilir.\n")

    # Eksik modaliteler (gait kamera arızalı)
    available_modalities = {
        'face': True,
        'fingerprint': True,
        'iris': False,       # Iris okuyucu arızalı
        'voice': True,
        'gait': False        # Kamera çalışmıyor
    }

    # Kalite skorları
    quality_scores = {
        'face': 0.90,        # İyi aydınlatma
        'fingerprint': 0.85,  # Temiz parmak
        'iris': 0.0,         # Mevcut değil
        'voice': 0.65,       # Hafif gürültülü
        'gait': 0.0          # Mevcut değil
    }

    print("Modalite Durumu:")
    for mod in modalities:
        status = "✓ Mevcut" if available_modalities[mod] else "✗ Mevcut değil"
        quality = quality_scores[mod]
        quality_str = f"(Kalite: {quality:.2f})" if available_modalities[mod] else ""
        print(f"  {mod:12s}: {status:20s} {quality_str}")

    # Sadece mevcut modalitelerle graf oluştur
    active_modalities = [m for m in modalities if available_modalities[m]]

    builder = ModalityGraphBuilder(
        modalities=active_modalities,
        edge_strategy='fully_connected',
        use_quality_scores=True
    )

    # Kalite tabanlı edge weights
    edge_weights = []
    for i in range(builder.edge_index.shape[1]):
        src_idx = builder.edge_index[0, i].item()
        dst_idx = builder.edge_index[1, i].item()

        src_mod = active_modalities[src_idx]
        dst_mod = active_modalities[dst_idx]

        weight = (quality_scores[src_mod] + quality_scores[dst_mod]) / 2
        edge_weights.append(weight)

    edge_weights = torch.tensor(edge_weights)

    # Görselleştir
    viz = GraphVisualizer(
        modalities=active_modalities,
        edge_index=builder.edge_index,
        edge_weights=edge_weights,
        style='presentation'
    )

    fig = viz.plot(
        layout='spring',
        show_edge_weights=True,
        title='Airport Security System (Missing Modalities)',
        save_path='outputs/graph_real_world_scenario.png'
    )

    plt.show()

    print(f"\nAktif modaliteler: {active_modalities}")
    print(f"Toplam bağlantı sayısı: {builder.edge_index.shape[1]}")
    print("\n✓ Gerçek dünya senaryosu grafiği kaydedildi: outputs/graph_real_world_scenario.png")


def main():
    parser = argparse.ArgumentParser(
        description='GNN Graph Structure Visualization Examples'
    )
    parser.add_argument(
        '--example',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help='Örnek numarası (0 = hepsi)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Çıktı klasörü'
    )
    args = parser.parse_args()

    # Çıktı klasörü oluştur
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nÇıktı klasörü: {output_dir.absolute()}")

    # Global output path ayarla
    import os
    os.chdir(Path(__file__).parent.parent)

    examples = {
        1: example_1_basic_graph_structure,
        2: example_2_edge_strategies_comparison,
        3: example_3_adaptive_edge_weights,
        4: example_4_attention_weights_heatmap,
        5: example_5_quality_aware_graph,
        6: example_6_real_world_scenario
    }

    if args.example == 0:
        # Tüm örnekleri çalıştır
        print("\n" + "="*80)
        print("TÜM ÖRNEKLER ÇALIŞTIRILIYOR")
        print("="*80)
        for example_func in examples.values():
            example_func()
            print("\n" + "-"*80)
    else:
        # Belirli bir örneği çalıştır
        examples[args.example]()

    print("\n" + "="*80)
    print("GÖRSELLEŞTIRME TAMAMLANDI!")
    print("="*80)
    print(f"\nTüm grafikler '{args.output}/' klasörüne kaydedildi.")
    print("\nDikkat: Bu örnekler simüle edilmiş verilerle çalışır.")
    print("Gerçek model ile kullanmak için train.py veya evaluate.py kullanın.")


if __name__ == '__main__':
    main()
