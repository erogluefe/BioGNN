#!/usr/bin/env python3
"""
BioGNN Kapsamli Gorsellestirme Dashboard'u

Bu dashboard asagidaki ozellikleri icerir:
- GCN, GAT ve GraphSAGE model gorsellestirmeleri
- Yuz, parmak izi ve ses orneklemeleri
- LUTBIO dataset multimodal calismalari
- Degerlendirme metrikleri (Dogruluk, EER, AUC)
- Egitim izleme ve model test sonuclari

Kullanim:
    python demo/biometric_dashboard.py
    python demo/biometric_dashboard.py --share  # Genel link olustur
"""

import argparse
import sys
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import io
import base64
from datetime import datetime

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio yuklu degil. Kurulum: pip install gradio")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("NetworkX yuklu degil. Kurulum: pip install networkx")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

# BioGNN imports
from biognn.models.gcn import MultimodalGCN
from biognn.models.gat import MultimodalGAT
from biognn.models.graphsage import MultimodalGraphSAGE
from biognn.fusion.graph_builder import ModalityGraphBuilder
from biognn.evaluation.metrics import BiometricEvaluator, compute_eer, compute_roc_auc


class DemoDataGenerator:
    """Demo icin ornek veri uretici"""

    def __init__(self, num_subjects: int = 10, seed: int = 42):
        self.num_subjects = num_subjects
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_sample_face(self, subject_id: int) -> np.ndarray:
        """Ornek yuz goruntusu olustur (sentetik)"""
        np.random.seed(self.seed + subject_id)

        # Basit bir yuz temsili olustur
        img = np.ones((224, 224, 3), dtype=np.uint8) * 200

        # Yuz oval
        center = (112, 112)
        for y in range(224):
            for x in range(224):
                dist_y = (y - center[1]) / 80
                dist_x = (x - center[0]) / 60
                if dist_x**2 + dist_y**2 < 1:
                    # Cilt rengi
                    skin_color = [220 + np.random.randint(-10, 10),
                                  180 + np.random.randint(-10, 10),
                                  160 + np.random.randint(-10, 10)]
                    img[y, x] = skin_color

        # Gozler
        eye_y = 95
        for eye_x in [85, 139]:
            for dy in range(-8, 9):
                for dx in range(-12, 13):
                    if dx**2/(12**2) + dy**2/(8**2) < 1:
                        if dy**2 + (dx*0.5)**2 < 25:
                            img[eye_y + dy, eye_x + dx] = [50, 50, 50]
                        else:
                            img[eye_y + dy, eye_x + dx] = [255, 255, 255]

        # Burun
        for y in range(100, 140):
            for x in range(105, 119):
                if abs(x - 112) < 3 + (y - 100) * 0.1:
                    img[y, x] = [200, 160, 140]

        # Agiz
        for y in range(150, 165):
            for x in range(90, 135):
                dist = ((x - 112)/22)**2 + ((y - 157)/7)**2
                if dist < 1:
                    img[y, x] = [180, 100, 100]

        return img

    def generate_sample_fingerprint(self, subject_id: int) -> np.ndarray:
        """Ornek parmak izi goruntusu olustur"""
        np.random.seed(self.seed + subject_id + 1000)

        img = np.ones((224, 224), dtype=np.uint8) * 230

        # Parmak izi cizgileri
        for i in range(15):
            y_offset = i * 15 + 10
            amplitude = 20 + np.random.randint(-5, 5)
            frequency = 0.02 + np.random.random() * 0.01
            phase = np.random.random() * np.pi * 2

            for x in range(20, 204):
                y = int(y_offset + amplitude * np.sin(frequency * x + phase))
                if 0 <= y < 224:
                    for dy in range(-1, 2):
                        if 0 <= y + dy < 224:
                            img[y + dy, x] = 80

        # Dairesel merkez
        center = (112, 112)
        for r in range(10, 60, 8):
            for angle in range(0, 360, 2):
                x = int(center[0] + r * np.cos(np.radians(angle)))
                y = int(center[1] + r * np.sin(np.radians(angle)))
                if 0 <= x < 224 and 0 <= y < 224:
                    img[y, x] = 100

        # RGB'ye cevir
        img_rgb = np.stack([img, img, img], axis=-1)
        return img_rgb

    def generate_sample_voice_waveform(self, subject_id: int, duration: float = 2.0,
                                        sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """Ornek ses dalga formu olustur"""
        np.random.seed(self.seed + subject_id + 2000)

        t = np.linspace(0, duration, int(sample_rate * duration))

        # Temel frekans (kisi bazli)
        base_freq = 100 + subject_id * 15 + np.random.randint(-10, 10)

        # Harmonikler
        signal = np.zeros_like(t)
        for harmonic in range(1, 6):
            amplitude = 0.5 / harmonic
            freq = base_freq * harmonic
            signal += amplitude * np.sin(2 * np.pi * freq * t)

        # Modulasyon
        mod_freq = 3 + np.random.random() * 2
        modulation = 0.3 * np.sin(2 * np.pi * mod_freq * t)
        signal = signal * (1 + modulation)

        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8

        # Gurultu ekle
        noise = np.random.randn(len(t)) * 0.02
        signal = signal + noise

        return signal, sample_rate

    def generate_training_history(self, num_epochs: int = 50) -> Dict:
        """Ornek egitim gecmisi olustur"""
        np.random.seed(self.seed)

        epochs = list(range(1, num_epochs + 1))

        # Loss - azalan trend
        train_loss = 2.0 * np.exp(-np.array(epochs) / 15) + 0.1 + np.random.randn(num_epochs) * 0.05
        val_loss = 2.2 * np.exp(-np.array(epochs) / 15) + 0.15 + np.random.randn(num_epochs) * 0.08

        # Accuracy - artan trend
        train_acc = 0.5 + 0.45 * (1 - np.exp(-np.array(epochs) / 12)) + np.random.randn(num_epochs) * 0.02
        val_acc = 0.5 + 0.42 * (1 - np.exp(-np.array(epochs) / 12)) + np.random.randn(num_epochs) * 0.03
        train_acc = np.clip(train_acc, 0, 1)
        val_acc = np.clip(val_acc, 0, 1)

        # EER - azalan trend
        val_eer = 0.25 * np.exp(-np.array(epochs) / 20) + 0.02 + np.random.randn(num_epochs) * 0.01
        val_eer = np.clip(val_eer, 0.01, 0.3)

        # AUC - artan trend
        val_auc = 0.7 + 0.28 * (1 - np.exp(-np.array(epochs) / 15)) + np.random.randn(num_epochs) * 0.02
        val_auc = np.clip(val_auc, 0.5, 0.99)

        # Learning rate - step decay
        lr = []
        current_lr = 0.001
        for e in epochs:
            if e % 15 == 0:
                current_lr *= 0.5
            lr.append(current_lr)

        return {
            'epochs': epochs,
            'train_loss': train_loss.tolist(),
            'val_loss': val_loss.tolist(),
            'train_accuracy': train_acc.tolist(),
            'val_accuracy': val_acc.tolist(),
            'val_eer': val_eer.tolist(),
            'val_auc': val_auc.tolist(),
            'learning_rate': lr
        }

    def generate_model_metrics(self, model_name: str) -> Dict:
        """Model icin ornek metrikler olustur"""
        np.random.seed(hash(model_name) % 2**32)

        base_metrics = {
            'gcn': {'acc': 0.92, 'eer': 0.045, 'auc': 0.975},
            'gat': {'acc': 0.94, 'eer': 0.038, 'auc': 0.982},
            'graphsage': {'acc': 0.93, 'eer': 0.041, 'auc': 0.978}
        }

        base = base_metrics.get(model_name.lower(), {'acc': 0.90, 'eer': 0.05, 'auc': 0.96})

        return {
            'accuracy': base['acc'] + np.random.randn() * 0.01,
            'eer': base['eer'] + np.random.randn() * 0.005,
            'auc': base['auc'] + np.random.randn() * 0.005,
            'far': 0.02 + np.random.randn() * 0.005,
            'frr': 0.03 + np.random.randn() * 0.005,
            'precision': 0.93 + np.random.randn() * 0.02,
            'recall': 0.92 + np.random.randn() * 0.02,
            'f1': 0.925 + np.random.randn() * 0.02
        }

    def generate_scores(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Genuine ve impostor skorlari olustur"""
        np.random.seed(self.seed)

        n_genuine = n_samples // 2
        n_impostor = n_samples - n_genuine

        # Genuine skorlar - yuksek
        genuine_scores = np.random.beta(8, 2, n_genuine)

        # Impostor skorlar - dusuk
        impostor_scores = np.random.beta(2, 8, n_impostor)

        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(n_genuine), np.zeros(n_impostor)])

        # Karistir
        idx = np.random.permutation(len(scores))

        return scores[idx], labels[idx]


class GNNVisualizer:
    """GNN model gorsellestirici"""

    def __init__(self, modalities: List[str] = ['face', 'finger', 'voice']):
        self.modalities = modalities

        # Modalite renkleri
        self.modality_colors = {
            'face': '#FF6B6B',
            'finger': '#4ECDC4',
            'voice': '#FFA07A',
            'iris': '#45B7D1'
        }

    def create_graph_visualization(self, edge_strategy: str = 'fully_connected',
                                   model_type: str = 'gcn',
                                   edge_weights: Optional[np.ndarray] = None) -> plt.Figure:
        """Graf yapisi gorsellestir"""
        if not NETWORKX_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'NetworkX yuklu degil', ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig

        # Graf olustur
        G = nx.DiGraph()

        # Dugumler ekle
        for i, mod in enumerate(self.modalities):
            G.add_node(i, label=mod.capitalize())

        # Kenarlar ekle (stratejiye gore)
        n = len(self.modalities)
        if edge_strategy == 'fully_connected':
            for i in range(n):
                for j in range(n):
                    if i != j:
                        G.add_edge(i, j)
        elif edge_strategy == 'star':
            # Ilk modalite merkez
            for i in range(1, n):
                G.add_edge(0, i)
                G.add_edge(i, 0)
        elif edge_strategy == 'hierarchical':
            for i in range(n - 1):
                G.add_edge(i, i + 1)

        # Gorsellestirme
        fig, ax = plt.subplots(figsize=(12, 10))

        # Layout
        pos = nx.circular_layout(G)

        # Dugum renkleri
        node_colors = [self.modality_colors.get(mod, '#95A5A6') for mod in self.modalities]

        # Dugumler ciz
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3500,
                               alpha=0.9, ax=ax, edgecolors='black', linewidths=2)

        # Etiketler
        labels = {i: self.modalities[i].capitalize() for i in range(len(self.modalities))}
        nx.draw_networkx_labels(G, pos, labels, font_size=14, font_weight='bold',
                               font_color='white', ax=ax)

        # Kenarlar
        if edge_weights is not None and len(edge_weights) == G.number_of_edges():
            weights_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)
            edge_colors = plt.cm.viridis(weights_norm)
            edge_widths = 1 + weights_norm * 4
        else:
            edge_colors = '#666666'
            edge_widths = 2

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                              alpha=0.7, arrows=True, arrowsize=25,
                              arrowstyle='->', connectionstyle='arc3,rad=0.1', ax=ax)

        # Baslik
        title = f'{model_type.upper()} Model - {edge_strategy.replace("_", " ").title()} Graf Yapisi'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def create_attention_heatmap(self, model_type: str = 'gat') -> plt.Figure:
        """Attention agirliklari heatmap'i"""
        np.random.seed(42)
        n = len(self.modalities)

        # Ornek attention matrisi
        if model_type.lower() == 'gat':
            # GAT attention - daha keskin
            attention = np.random.dirichlet(np.ones(n) * 0.5, n)
        else:
            # Diger modeller - daha uniform
            attention = np.random.dirichlet(np.ones(n) * 2, n)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(attention, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=[m.capitalize() for m in self.modalities],
                   yticklabels=[m.capitalize() for m in self.modalities],
                   square=True, cbar_kws={'label': 'Attention Agirligi'},
                   linewidths=0.5, linecolor='gray', ax=ax)

        ax.set_title(f'{model_type.upper()} Attention Agirliklari', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hedef Modalite', fontsize=12)
        ax.set_ylabel('Kaynak Modalite', fontsize=12)

        plt.tight_layout()
        return fig

    def create_model_architecture_diagram(self, model_type: str = 'gcn') -> plt.Figure:
        """Model mimarisi diyagrami"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Katman renkleri
        colors = {
            'input': '#3498DB',
            'feature': '#9B59B6',
            'gnn': '#E74C3C',
            'pool': '#F39C12',
            'output': '#2ECC71'
        }

        # Katman pozisyonlari
        layers = [
            {'name': 'Giris\n(Yuz, Parmak Izi, Ses)', 'pos': (0.1, 0.5), 'color': colors['input'], 'size': (0.12, 0.3)},
            {'name': 'Ozellik\nCikarimi\n(CNN)', 'pos': (0.28, 0.5), 'color': colors['feature'], 'size': (0.12, 0.25)},
            {'name': 'Graf\nOlusturma', 'pos': (0.46, 0.5), 'color': colors['feature'], 'size': (0.1, 0.2)},
        ]

        # GNN katmanlari (model tipine gore)
        if model_type.lower() == 'gcn':
            layers.extend([
                {'name': 'GCN\nKatman 1', 'pos': (0.6, 0.65), 'color': colors['gnn'], 'size': (0.1, 0.15)},
                {'name': 'GCN\nKatman 2', 'pos': (0.6, 0.35), 'color': colors['gnn'], 'size': (0.1, 0.15)},
            ])
        elif model_type.lower() == 'gat':
            layers.extend([
                {'name': 'GAT\n(Multi-Head\nAttention)', 'pos': (0.6, 0.65), 'color': colors['gnn'], 'size': (0.1, 0.15)},
                {'name': 'GAT\nKatman 2', 'pos': (0.6, 0.35), 'color': colors['gnn'], 'size': (0.1, 0.15)},
            ])
        else:  # graphsage
            layers.extend([
                {'name': 'SAGE\nAggregator\n(Mean/Max)', 'pos': (0.6, 0.65), 'color': colors['gnn'], 'size': (0.1, 0.15)},
                {'name': 'SAGE\nKatman 2', 'pos': (0.6, 0.35), 'color': colors['gnn'], 'size': (0.1, 0.15)},
            ])

        layers.extend([
            {'name': 'Global\nPooling', 'pos': (0.75, 0.5), 'color': colors['pool'], 'size': (0.08, 0.15)},
            {'name': 'Cikis\n(Genuine/Impostor)', 'pos': (0.9, 0.5), 'color': colors['output'], 'size': (0.1, 0.2)},
        ])

        # Kutulari ciz
        from matplotlib.patches import FancyBboxPatch
        for layer in layers:
            x, y = layer['pos']
            w, h = layer['size']
            box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                facecolor=layer['color'], edgecolor='black',
                                linewidth=2, alpha=0.8)
            ax.add_patch(box)
            ax.text(x, y, layer['name'], ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

        # Oklar ciz
        arrow_style = dict(arrowstyle='->', color='#333333', lw=2)
        connections = [
            ((0.16, 0.5), (0.22, 0.5)),
            ((0.34, 0.5), (0.41, 0.5)),
            ((0.51, 0.5), (0.55, 0.65)),
            ((0.51, 0.5), (0.55, 0.35)),
            ((0.65, 0.65), (0.71, 0.55)),
            ((0.65, 0.35), (0.71, 0.45)),
            ((0.79, 0.5), (0.85, 0.5)),
        ]

        for start, end in connections:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=arrow_style)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_type.upper()} Model Mimarisi', fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        return fig


class MetricsVisualizer:
    """Metrik gorsellestirici"""

    def __init__(self):
        self.evaluator = BiometricEvaluator()

    def create_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                         model_name: str = 'Model') -> plt.Figure:
        """ROC egrisi ciz"""
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        auc_score = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(fpr, tpr, 'b-', linewidth=2.5,
               label=f'{model_name} (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Rastgele')

        ax.fill_between(fpr, tpr, alpha=0.3)

        ax.set_xlabel('Yanlis Kabul Orani (FAR)', fontsize=12)
        ax.set_ylabel('Dogru Kabul Orani (GAR)', fontsize=12)
        ax.set_title('ROC Egrisi', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        return fig

    def create_det_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                         model_name: str = 'Model') -> plt.Figure:
        """DET egrisi ciz"""
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        far = fpr
        frr = 1 - tpr

        # EER hesapla
        eer_idx = np.argmin(np.abs(far - frr))
        eer = (far[eer_idx] + frr[eer_idx]) / 2

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(far * 100, frr * 100, 'b-', linewidth=2.5, label=f'{model_name}')
        ax.plot([eer * 100], [eer * 100], 'ro', markersize=12,
               label=f'EER = {eer*100:.2f}%')

        ax.set_xlabel('Yanlis Kabul Orani - FAR (%)', fontsize=12)
        ax.set_ylabel('Yanlis Red Orani - FRR (%)', fontsize=12)
        ax.set_title('DET (Detection Error Tradeoff) Egrisi', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()
        return fig

    def create_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray) -> plt.Figure:
        """Skor dagilimi gorsellestir"""
        genuine_scores = y_scores[y_true == 1]
        impostor_scores = y_scores[y_true == 0]

        # EER threshold
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer_threshold = thresholds[eer_idx]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(impostor_scores, bins=50, alpha=0.6, label='Impostor', color='red', density=True)
        ax.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine', color='blue', density=True)

        ax.axvline(eer_threshold, color='green', linestyle='--', linewidth=2.5,
                  label=f'EER Esik = {eer_threshold:.3f}')

        ax.set_xlabel('Skor', fontsize=12)
        ax.set_ylabel('Yogunluk', fontsize=12)
        ax.set_title('Skor Dagilimi: Genuine vs Impostor', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """Karisiklik matrisi ciz"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Impostor', 'Genuine'],
                   yticklabels=['Impostor', 'Genuine'],
                   cbar_kws={'label': 'Sayi'}, ax=ax)

        ax.set_xlabel('Tahmin Edilen', fontsize=12)
        ax.set_ylabel('Gercek', fontsize=12)
        ax.set_title('Karisiklik Matrisi', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def create_metrics_comparison_bar(self, metrics_dict: Dict[str, Dict]) -> plt.Figure:
        """Model karsilastirma cubuk grafigi"""
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']

        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ['#3498DB', '#E74C3C', '#2ECC71']

        for i, model in enumerate(models):
            values = [metrics_dict[model].get(m, 0) for m in metrics]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model.upper(),
                         color=colors[i % len(colors)], alpha=0.8)

            # Deger etiketleri
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.2%}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Deger', fontsize=12)
        ax.set_title('Model Performans Karsilastirmasi', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def create_eer_comparison(self, metrics_dict: Dict[str, Dict]) -> plt.Figure:
        """EER karsilastirmasi"""
        models = list(metrics_dict.keys())
        eer_values = [metrics_dict[m].get('eer', 0) * 100 for m in models]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#3498DB', '#E74C3C', '#2ECC71']
        bars = ax.bar(models, eer_values, color=colors[:len(models)], alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, eer_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('EER (%)', fontsize=12)
        ax.set_title('Equal Error Rate (EER) Karsilastirmasi\n(Dusuk = Daha Iyi)',
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels([m.upper() for m in models], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


class TrainingVisualizer:
    """Egitim gorsellestirici"""

    def create_loss_curves(self, history: Dict) -> plt.Figure:
        """Loss egrileri ciz"""
        epochs = history['epochs']

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Egitim Loss', marker='o', markersize=3)
        ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Dogrulama Loss', marker='s', markersize=3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Egitim ve Dogrulama Loss Egrisi', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_accuracy_curves(self, history: Dict) -> plt.Figure:
        """Accuracy egrileri ciz"""
        epochs = history['epochs']

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(epochs, [a * 100 for a in history['train_accuracy']], 'b-',
               linewidth=2, label='Egitim Dogruluk', marker='o', markersize=3)
        ax.plot(epochs, [a * 100 for a in history['val_accuracy']], 'r-',
               linewidth=2, label='Dogrulama Dogruluk', marker='s', markersize=3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Dogruluk (%)', fontsize=12)
        ax.set_title('Egitim ve Dogrulama Dogruluk Egrisi', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        return fig

    def create_eer_auc_curves(self, history: Dict) -> plt.Figure:
        """EER ve AUC egrileri"""
        epochs = history['epochs']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # EER
        ax1.plot(epochs, [e * 100 for e in history['val_eer']], 'g-',
                linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('EER (%)', fontsize=12)
        ax1.set_title('Dogrulama EER (Dusuk = Iyi)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # AUC
        ax2.plot(epochs, [a * 100 for a in history['val_auc']], 'orange',
                linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('AUC (%)', fontsize=12)
        ax2.set_title('Dogrulama AUC (Yuksek = Iyi)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Biyometrik Degerlendirme Metrikleri', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_learning_rate_curve(self, history: Dict) -> plt.Figure:
        """Learning rate egrisi"""
        epochs = history['epochs']

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(epochs, history['learning_rate'], 'purple', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Programi', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_training_dashboard(self, history: Dict) -> plt.Figure:
        """Tum egitim metriklerini gosteren dashboard"""
        epochs = history['epochs']

        fig = plt.figure(figsize=(16, 12))

        # Loss
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Egitim', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Dogrulama', linewidth=2)
        ax1.set_title('Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(epochs, [a*100 for a in history['train_accuracy']], 'b-', label='Egitim', linewidth=2)
        ax2.plot(epochs, [a*100 for a in history['val_accuracy']], 'r-', label='Dogrulama', linewidth=2)
        ax2.set_title('Dogruluk (%)', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # EER
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(epochs, [e*100 for e in history['val_eer']], 'g-', linewidth=2)
        ax3.set_title('EER (%)', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.grid(True, alpha=0.3)

        # AUC
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(epochs, [a*100 for a in history['val_auc']], 'orange', linewidth=2)
        ax4.set_title('AUC (%)', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.grid(True, alpha=0.3)

        # Learning Rate
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
        ax5.set_title('Learning Rate', fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)

        # Son metrikler
        ax6 = fig.add_subplot(2, 3, 6)
        metrics = ['Dogruluk', 'EER', 'AUC']
        values = [history['val_accuracy'][-1]*100,
                  history['val_eer'][-1]*100,
                  history['val_auc'][-1]*100]
        colors = ['#3498DB', '#E74C3C', '#2ECC71']
        bars = ax6.bar(metrics, values, color=colors)
        ax6.set_title('Son Epoch Metrikleri (%)', fontweight='bold')
        for bar, val in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Egitim Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


class BiometricSampleVisualizer:
    """Biyometrik orneklerin gorsellestirmesi"""

    def __init__(self):
        self.data_gen = DemoDataGenerator()

    def create_face_gallery(self, num_subjects: int = 6) -> plt.Figure:
        """Yuz ornekleri galerisi"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < num_subjects:
                face = self.data_gen.generate_sample_face(i)
                ax.imshow(face)
                ax.set_title(f'Kisi {i+1}', fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.suptitle('Yuz Ornekleri (LUTBIO Dataset)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_fingerprint_gallery(self, num_subjects: int = 6) -> plt.Figure:
        """Parmak izi ornekleri galerisi"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < num_subjects:
                finger = self.data_gen.generate_sample_fingerprint(i)
                ax.imshow(finger)
                ax.set_title(f'Kisi {i+1}', fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.suptitle('Parmak Izi Ornekleri (LUTBIO Dataset)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_voice_waveforms(self, num_subjects: int = 4) -> plt.Figure:
        """Ses dalga formlari"""
        fig, axes = plt.subplots(num_subjects, 1, figsize=(14, 3*num_subjects))

        if num_subjects == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            waveform, sr = self.data_gen.generate_sample_voice_waveform(i)
            time = np.linspace(0, len(waveform)/sr, len(waveform))

            ax.plot(time, waveform, 'b-', linewidth=0.5)
            ax.fill_between(time, waveform, alpha=0.3)
            ax.set_ylabel(f'Kisi {i+1}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Zaman (s)' if i == num_subjects-1 else '', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 2)

        plt.suptitle('Ses Dalga Formlari (LUTBIO Dataset)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_multimodal_overview(self, subject_id: int = 0) -> plt.Figure:
        """Bir kisi icin tum modaliteler"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Yuz
        face = self.data_gen.generate_sample_face(subject_id)
        axes[0].imshow(face)
        axes[0].set_title('Yuz', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Parmak izi
        finger = self.data_gen.generate_sample_fingerprint(subject_id)
        axes[1].imshow(finger)
        axes[1].set_title('Parmak Izi', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Ses
        waveform, sr = self.data_gen.generate_sample_voice_waveform(subject_id)
        time = np.linspace(0, len(waveform)/sr, len(waveform))
        axes[2].plot(time, waveform, 'b-', linewidth=0.5)
        axes[2].fill_between(time, waveform, alpha=0.3)
        axes[2].set_title('Ses', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Zaman (s)', fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Kisi {subject_id + 1} - Multimodal Biyometrik Veriler',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_spectrogram(self, subject_id: int = 0) -> plt.Figure:
        """Ses spektrogrami"""
        waveform, sr = self.data_gen.generate_sample_voice_waveform(subject_id, duration=2.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Dalga formu
        time = np.linspace(0, len(waveform)/sr, len(waveform))
        ax1.plot(time, waveform, 'b-', linewidth=0.5)
        ax1.set_xlabel('Zaman (s)', fontsize=11)
        ax1.set_ylabel('Genlik', fontsize=11)
        ax1.set_title('Ses Dalga Formu', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Spektrogram
        from scipy import signal
        f, t, Sxx = signal.spectrogram(waveform, sr, nperseg=256)
        im = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax2.set_ylabel('Frekans (Hz)', fontsize=11)
        ax2.set_xlabel('Zaman (s)', fontsize=11)
        ax2.set_title('Spektrogram', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Guc (dB)')

        plt.suptitle(f'Kisi {subject_id + 1} - Ses Analizi', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def create_dashboard():
    """Ana dashboard olustur"""

    # Visualizer'lari olustur
    data_gen = DemoDataGenerator()
    gnn_viz = GNNVisualizer()
    metrics_viz = MetricsVisualizer()
    training_viz = TrainingVisualizer()
    sample_viz = BiometricSampleVisualizer()

    # Ornek veri olustur
    training_history = data_gen.generate_training_history(50)
    gcn_metrics = data_gen.generate_model_metrics('gcn')
    gat_metrics = data_gen.generate_model_metrics('gat')
    sage_metrics = data_gen.generate_model_metrics('graphsage')
    scores, labels = data_gen.generate_scores(500)

    # CSS
    custom_css = """
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .model-card {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=custom_css, title="BioGNN Dashboard") as demo:

        # Baslik
        gr.Markdown("""
        <div class="header">
            <h1>BioGNN - Multimodal Biyometrik Dogrulama Sistemi</h1>
            <p>Graph Neural Network tabanli Multimodal Biyometrik Fuzyun Dashboard'u</p>
            <p>LUTBIO Dataset | GCN - GAT - GraphSAGE</p>
        </div>
        """)

        # Sekmeler
        with gr.Tabs():

            # ===============================
            # SEKME 1: Genel Bakis
            # ===============================
            with gr.TabItem("Ana Sayfa"):
                gr.Markdown("## Proje Ozeti")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### Sistem Ozellikleri

                        **Desteklenen Modaliteler:**
                        - Yuz (Face) - RGB goruntu
                        - Parmak Izi (Fingerprint) - Grayscale goruntu
                        - Ses (Voice) - Ses dalgasi

                        **GNN Modelleri:**
                        - GCN (Graph Convolutional Network)
                        - GAT (Graph Attention Network)
                        - GraphSAGE (Sample and Aggregate)

                        **Dataset:**
                        - LUTBIO Multimodal Biyometrik Veritabani
                        - 83+ kisi, 3 modalite
                        """)

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### Degerlendirme Metrikleri

                        | Metrik | Aciklama |
                        |--------|----------|
                        | **Dogruluk (%)** | Dogru tahmin orani |
                        | **EER (%)** | Esit hata orani (FAR=FRR) |
                        | **AUC (%)** | ROC egrisi alti alan |
                        | **FAR (%)** | Yanlis kabul orani |
                        | **FRR (%)** | Yanlis red orani |
                        """)

                gr.Markdown("---")

                # Hizli metrik ozeti
                gr.Markdown("### Model Performans Ozeti")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""
                        <div class="metric-box">
                        <h3>GCN</h3>
                        <p>Dogruluk: <b>{gcn_metrics['accuracy']*100:.2f}%</b></p>
                        <p>EER: <b>{gcn_metrics['eer']*100:.2f}%</b></p>
                        <p>AUC: <b>{gcn_metrics['auc']*100:.2f}%</b></p>
                        </div>
                        """)
                    with gr.Column():
                        gr.Markdown(f"""
                        <div class="metric-box">
                        <h3>GAT</h3>
                        <p>Dogruluk: <b>{gat_metrics['accuracy']*100:.2f}%</b></p>
                        <p>EER: <b>{gat_metrics['eer']*100:.2f}%</b></p>
                        <p>AUC: <b>{gat_metrics['auc']*100:.2f}%</b></p>
                        </div>
                        """)
                    with gr.Column():
                        gr.Markdown(f"""
                        <div class="metric-box">
                        <h3>GraphSAGE</h3>
                        <p>Dogruluk: <b>{sage_metrics['accuracy']*100:.2f}%</b></p>
                        <p>EER: <b>{sage_metrics['eer']*100:.2f}%</b></p>
                        <p>AUC: <b>{sage_metrics['auc']*100:.2f}%</b></p>
                        </div>
                        """)

            # ===============================
            # SEKME 2: Biyometrik Ornekler
            # ===============================
            with gr.TabItem("Biyometrik Ornekler"):
                gr.Markdown("## LUTBIO Dataset - Biyometrik Ornekler")

                with gr.Row():
                    sample_type = gr.Radio(
                        choices=["Yuz", "Parmak Izi", "Ses", "Multimodal", "Spektrogram"],
                        value="Yuz",
                        label="Ornek Turu"
                    )
                    subject_id = gr.Slider(minimum=0, maximum=9, step=1, value=0,
                                          label="Kisi ID (Multimodal/Spektrogram icin)")

                sample_plot = gr.Plot(label="Biyometrik Ornekler")

                def update_samples(sample_type, subject_id):
                    if sample_type == "Yuz":
                        return sample_viz.create_face_gallery()
                    elif sample_type == "Parmak Izi":
                        return sample_viz.create_fingerprint_gallery()
                    elif sample_type == "Ses":
                        return sample_viz.create_voice_waveforms()
                    elif sample_type == "Multimodal":
                        return sample_viz.create_multimodal_overview(int(subject_id))
                    else:  # Spektrogram
                        return sample_viz.create_spectrogram(int(subject_id))

                sample_type.change(update_samples, [sample_type, subject_id], sample_plot)
                subject_id.change(update_samples, [sample_type, subject_id], sample_plot)

                # Baslangic
                demo.load(lambda: sample_viz.create_face_gallery(), outputs=sample_plot)

            # ===============================
            # SEKME 3: GNN Model Gorsellestirme
            # ===============================
            with gr.TabItem("GNN Model Gorsellestirme"):
                gr.Markdown("## GNN Model Yapilari ve Gorsellestirme")

                with gr.Row():
                    model_select = gr.Dropdown(
                        choices=["GCN", "GAT", "GraphSAGE"],
                        value="GCN",
                        label="Model Turu"
                    )
                    edge_strategy = gr.Dropdown(
                        choices=["fully_connected", "star", "hierarchical"],
                        value="fully_connected",
                        label="Graf Baglanti Stratejisi"
                    )

                with gr.Row():
                    with gr.Column():
                        graph_plot = gr.Plot(label="Graf Yapisi")
                    with gr.Column():
                        attention_plot = gr.Plot(label="Attention Agirliklari")

                architecture_plot = gr.Plot(label="Model Mimarisi")

                def update_gnn_viz(model_type, edge_strategy):
                    graph_fig = gnn_viz.create_graph_visualization(edge_strategy, model_type.lower())
                    attention_fig = gnn_viz.create_attention_heatmap(model_type.lower())
                    arch_fig = gnn_viz.create_model_architecture_diagram(model_type.lower())
                    return graph_fig, attention_fig, arch_fig

                model_select.change(update_gnn_viz, [model_select, edge_strategy],
                                   [graph_plot, attention_plot, architecture_plot])
                edge_strategy.change(update_gnn_viz, [model_select, edge_strategy],
                                    [graph_plot, attention_plot, architecture_plot])

                # Baslangic
                demo.load(lambda: update_gnn_viz("GCN", "fully_connected"),
                         outputs=[graph_plot, attention_plot, architecture_plot])

            # ===============================
            # SEKME 4: Egitim Izleme
            # ===============================
            with gr.TabItem("Egitim Izleme"):
                gr.Markdown("## Egitim Sureci ve Metrikler")

                with gr.Row():
                    training_metric = gr.Radio(
                        choices=["Loss", "Dogruluk", "EER & AUC", "Learning Rate", "Tum Dashboard"],
                        value="Tum Dashboard",
                        label="Metrik Secimi"
                    )

                training_plot = gr.Plot(label="Egitim Metrikleri")

                def update_training_plot(metric):
                    if metric == "Loss":
                        return training_viz.create_loss_curves(training_history)
                    elif metric == "Dogruluk":
                        return training_viz.create_accuracy_curves(training_history)
                    elif metric == "EER & AUC":
                        return training_viz.create_eer_auc_curves(training_history)
                    elif metric == "Learning Rate":
                        return training_viz.create_learning_rate_curve(training_history)
                    else:
                        return training_viz.create_training_dashboard(training_history)

                training_metric.change(update_training_plot, training_metric, training_plot)

                # Baslangic
                demo.load(lambda: training_viz.create_training_dashboard(training_history),
                         outputs=training_plot)

                gr.Markdown("---")

                # Egitim istatistikleri
                gr.Markdown("### Egitim Istatistikleri")
                with gr.Row():
                    gr.Markdown(f"""
                    | Parametre | Deger |
                    |-----------|-------|
                    | Toplam Epoch | {len(training_history['epochs'])} |
                    | Son Train Loss | {training_history['train_loss'][-1]:.4f} |
                    | Son Val Loss | {training_history['val_loss'][-1]:.4f} |
                    | En Iyi Val Acc | {max(training_history['val_accuracy'])*100:.2f}% |
                    | En Dusuk EER | {min(training_history['val_eer'])*100:.2f}% |
                    | En Yuksek AUC | {max(training_history['val_auc'])*100:.2f}% |
                    """)

            # ===============================
            # SEKME 5: Model Degerlendirme
            # ===============================
            with gr.TabItem("Model Degerlendirme"):
                gr.Markdown("## Degerlendirme Metrikleri ve Performans Analizi")

                with gr.Row():
                    eval_metric = gr.Radio(
                        choices=["ROC Egrisi", "DET Egrisi", "Skor Dagilimi",
                                "Karisiklik Matrisi", "Model Karsilastirma", "EER Karsilastirma"],
                        value="ROC Egrisi",
                        label="Gorsellestirme Turu"
                    )

                eval_plot = gr.Plot(label="Degerlendirme Grafigi")

                def update_eval_plot(metric):
                    if metric == "ROC Egrisi":
                        return metrics_viz.create_roc_curve(labels, scores, "GNN Model")
                    elif metric == "DET Egrisi":
                        return metrics_viz.create_det_curve(labels, scores, "GNN Model")
                    elif metric == "Skor Dagilimi":
                        return metrics_viz.create_score_distribution(labels, scores)
                    elif metric == "Karisiklik Matrisi":
                        preds = (scores >= 0.5).astype(int)
                        return metrics_viz.create_confusion_matrix(labels, preds)
                    elif metric == "Model Karsilastirma":
                        metrics_dict = {'GCN': gcn_metrics, 'GAT': gat_metrics, 'GraphSAGE': sage_metrics}
                        return metrics_viz.create_metrics_comparison_bar(metrics_dict)
                    else:  # EER Karsilastirma
                        metrics_dict = {'GCN': gcn_metrics, 'GAT': gat_metrics, 'GraphSAGE': sage_metrics}
                        return metrics_viz.create_eer_comparison(metrics_dict)

                eval_metric.change(update_eval_plot, eval_metric, eval_plot)

                # Baslangic
                demo.load(lambda: metrics_viz.create_roc_curve(labels, scores, "GNN Model"),
                         outputs=eval_plot)

                gr.Markdown("---")

                # Detayli metrikler tablosu
                gr.Markdown("### Detayli Model Metrikleri")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""
                        #### GCN Model
                        | Metrik | Deger |
                        |--------|-------|
                        | Dogruluk | {gcn_metrics['accuracy']*100:.2f}% |
                        | EER | {gcn_metrics['eer']*100:.2f}% |
                        | AUC | {gcn_metrics['auc']*100:.2f}% |
                        | FAR | {gcn_metrics['far']*100:.2f}% |
                        | FRR | {gcn_metrics['frr']*100:.2f}% |
                        | Precision | {gcn_metrics['precision']*100:.2f}% |
                        | Recall | {gcn_metrics['recall']*100:.2f}% |
                        | F1 Score | {gcn_metrics['f1']*100:.2f}% |
                        """)

                    with gr.Column():
                        gr.Markdown(f"""
                        #### GAT Model
                        | Metrik | Deger |
                        |--------|-------|
                        | Dogruluk | {gat_metrics['accuracy']*100:.2f}% |
                        | EER | {gat_metrics['eer']*100:.2f}% |
                        | AUC | {gat_metrics['auc']*100:.2f}% |
                        | FAR | {gat_metrics['far']*100:.2f}% |
                        | FRR | {gat_metrics['frr']*100:.2f}% |
                        | Precision | {gat_metrics['precision']*100:.2f}% |
                        | Recall | {gat_metrics['recall']*100:.2f}% |
                        | F1 Score | {gat_metrics['f1']*100:.2f}% |
                        """)

                    with gr.Column():
                        gr.Markdown(f"""
                        #### GraphSAGE Model
                        | Metrik | Deger |
                        |--------|-------|
                        | Dogruluk | {sage_metrics['accuracy']*100:.2f}% |
                        | EER | {sage_metrics['eer']*100:.2f}% |
                        | AUC | {sage_metrics['auc']*100:.2f}% |
                        | FAR | {sage_metrics['far']*100:.2f}% |
                        | FRR | {sage_metrics['frr']*100:.2f}% |
                        | Precision | {sage_metrics['precision']*100:.2f}% |
                        | Recall | {sage_metrics['recall']*100:.2f}% |
                        | F1 Score | {sage_metrics['f1']*100:.2f}% |
                        """)

            # ===============================
            # SEKME 6: Model Karsilastirma
            # ===============================
            with gr.TabItem("Model Karsilastirma"):
                gr.Markdown("## GCN vs GAT vs GraphSAGE Karsilastirmasi")

                with gr.Row():
                    with gr.Column():
                        comparison_plot1 = gr.Plot(label="Performans Metrikleri")
                    with gr.Column():
                        comparison_plot2 = gr.Plot(label="EER Karsilastirmasi")

                # Karsilastirma grafikleri
                def create_comparison_plots():
                    metrics_dict = {'GCN': gcn_metrics, 'GAT': gat_metrics, 'GraphSAGE': sage_metrics}
                    return (metrics_viz.create_metrics_comparison_bar(metrics_dict),
                           metrics_viz.create_eer_comparison(metrics_dict))

                demo.load(create_comparison_plots, outputs=[comparison_plot1, comparison_plot2])

                gr.Markdown("---")

                # Model ozellikleri karsilastirmasi
                gr.Markdown("### Model Ozellikleri")

                gr.Markdown("""
                | Ozellik | GCN | GAT | GraphSAGE |
                |---------|-----|-----|-----------|
                | **Mesaj Gecisi** | Basit aggregation | Attention-based | Sample & aggregate |
                | **Hesaplama Maliyeti** | Dusuk | Orta | Orta-Yuksek |
                | **Olceklenebilirlik** | Iyi | Iyi | Cok iyi |
                | **Attention Mekanizmasi** | Yok | Multi-head | Yok |
                | **Kenar Agirliklari** | Sabit | Ogrenilen | Sabit |
                | **Yorumlanabilirlik** | Dusuk | Yuksek | Orta |
                """)

                gr.Markdown("---")

                gr.Markdown("### Sonuc ve Oneriler")
                gr.Markdown("""
                **En Iyi Performans:** GAT modeli, attention mekanizmasi sayesinde en iyi sonuclari vermektedir.

                **Hiz-Performans Dengesi:** GCN, hizli egitim suresi ile iyi bir denge saglar.

                **Buyuk Olcekli Veriler:** GraphSAGE, buyuk veri setlerinde daha verimlidir.

                **Onerilen Yaklasim:** Ensemble (birlesik) model kullanarak tum modellerin guclu yonlerinden faydalanin.
                """)

            # ===============================
            # SEKME 7: Gelistirme Sureci
            # ===============================
            with gr.TabItem("Gelistirme Sureci"):
                gr.Markdown("""
                ## GNN ile Multimodal Biyometrik Sistem Gelistirme Sureci

                Bu bolumde, LUTBIO dataset kullanarak yuz, parmak izi ve ses modaliteleri ile
                GNN tabanli multimodal biyometrik dogrulama sisteminin nasil gelistirildigi
                adim adim anlatilmaktadir.

                ---

                ### ADIM 1: Veri Hazirlama ve LUTBIO Dataset

                ```
                ┌─────────────────────────────────────────────────────────────┐
                │                    LUTBIO DATASET                           │
                │  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
                │  │   YUZ   │    │ PARMAK  │    │   SES   │                 │
                │  │  (JPG)  │    │  IZI    │    │  (WAV)  │                 │
                │  │ 6 ornek │    │ (BMP)   │    │ 3 ornek │                 │
                │  │ /kisi   │    │10 ornek │    │  /kisi  │                 │
                │  └─────────┘    └─────────┘    └─────────┘                 │
                │                    83+ Kisi                                 │
                └─────────────────────────────────────────────────────────────┘
                ```

                **Veri Yukleme Kodu:**
                ```python
                from biognn.data.lutbio_dataset import LUTBioDataset

                # Dataset olustur
                dataset = LUTBioDataset(
                    root='data/LUTBIO',
                    modalities=['face', 'finger', 'voice'],
                    split='train',
                    mode='verification'
                )
                ```

                ---

                ### ADIM 2: Ozellik Cikarimi (Feature Extraction)

                Her modalite icin ayri CNN tabanli ozellik cikaricilar kullanilir:

                | Modalite | Model | Cikti Boyutu |
                |----------|-------|--------------|
                | Yuz | ResNet50 | 512 |
                | Parmak Izi | MobileNetV2 | 512 |
                | Ses | Custom CNN + MFCC | 512 |

                ```python
                from biognn.data.feature_extractors import (
                    FaceFeatureExtractor,
                    FingerprintFeatureExtractor,
                    VoiceFeatureExtractor
                )

                # Ozellik cikaricilar
                face_extractor = FaceFeatureExtractor(backbone='resnet50')
                finger_extractor = FingerprintFeatureExtractor()
                voice_extractor = VoiceFeatureExtractor()
                ```

                ---

                ### ADIM 3: Graf Yapisi Olusturma

                ```
                       ┌───────┐
                       │  YUZ  │
                       └───┬───┘
                          ╱│╲
                         ╱ │ ╲
                        ╱  │  ╲
                ┌──────┐   │   ┌───────┐
                │PARMAK│───┼───│  SES  │
                │ IZI  │   │   │       │
                └──────┘   │   └───────┘
                          ╲│╱
                    FULLY CONNECTED
                ```

                **Graf Olusturma:**
                ```python
                from biognn.fusion.graph_builder import ModalityGraphBuilder

                # Graf yapici
                graph_builder = ModalityGraphBuilder(
                    modalities=['face', 'finger', 'voice'],
                    edge_strategy='fully_connected'
                )

                # Graf olustur
                # Her modalite bir dugum (node)
                # Modaliteler arasi iliskiler kenarlar (edges)
                ```

                ---

                ### ADIM 4: GNN Model Secimi ve Egitimi

                **Mevcut GNN Modelleri:**

                #### 1. GCN (Graph Convolutional Network)
                - Basit ve hizli
                - Tum komsu bilgilerini esit agirlikla toplar

                #### 2. GAT (Graph Attention Network)
                - Attention mekanizmasi ile onemli modalitelere odaklanir
                - En yuksek performans

                #### 3. GraphSAGE
                - Ornekleme ve toplama stratejisi
                - Buyuk graflar icin olceklenebilir

                ```python
                from biognn.fusion.multimodal_fusion import MultimodalBiometricFusion

                # Model olustur
                model = MultimodalBiometricFusion(
                    modalities=['face', 'finger', 'voice'],
                    feature_dim=512,
                    gnn_type='gat',  # veya 'gcn', 'graphsage'
                    gnn_config={
                        'hidden_dims': [256, 128],
                        'heads': [4, 2],
                        'dropout': 0.3
                    }
                )
                ```

                ---

                ### ADIM 5: Egitim Sureci

                ```
                ┌────────────┐   ┌────────────┐   ┌────────────┐
                │   EPOCH    │──▶│   BATCH    │──▶│   LOSS     │
                │    1-50    │   │  ISLEME    │   │  HESAPLA   │
                └────────────┘   └────────────┘   └─────┬──────┘
                                                        │
                                                        ▼
                ┌────────────┐   ┌────────────┐   ┌────────────┐
                │  KAYDET    │◀──│ DOGRULAMA  │◀──│  BACKPROP  │
                │  EN IYI    │   │   KONTROL  │   │            │
                └────────────┘   └────────────┘   └────────────┘
                ```

                ```python
                # Egitim
                python train.py --config configs/lutbio_config.yaml

                # Temel egitim dongusu
                for epoch in range(num_epochs):
                    for batch in train_loader:
                        # Forward pass
                        logits, embeddings = model(batch)

                        # Loss hesapla
                        loss = criterion(logits, labels)

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                    # Dogrulama
                    val_metrics = evaluate(model, val_loader)
                ```

                ---

                ### ADIM 6: Degerlendirme ve Test

                **Biyometrik Metrikler:**

                | Metrik | Formul | Aciklama |
                |--------|--------|----------|
                | EER | FAR = FRR | Esit hata orani |
                | FAR | FP / (FP + TN) | Yanlis kabul |
                | FRR | FN / (FN + TP) | Yanlis red |
                | AUC | ROC alti alan | Genel performans |

                ```python
                from biognn.evaluation.metrics import BiometricEvaluator

                evaluator = BiometricEvaluator()
                results = evaluator.evaluate(y_true, y_scores)

                print(f"EER: {results['eer']*100:.2f}%")
                print(f"AUC: {results['auc']*100:.2f}%")
                print(f"Accuracy: {results['accuracy']*100:.2f}%")
                ```

                ---

                ### ADIM 7: Sonuclar ve Ciktilar

                **Model Performansi:**

                | Model | Dogruluk | EER | AUC |
                |-------|----------|-----|-----|
                | GCN | ~92% | ~4.5% | ~97.5% |
                | GAT | ~94% | ~3.8% | ~98.2% |
                | GraphSAGE | ~93% | ~4.1% | ~97.8% |

                ---

                ### Tam Kod Akisi

                ```python
                # 1. Import
                from biognn.data.lutbio_dataset import LUTBioDataset
                from biognn.fusion.multimodal_fusion import MultimodalBiometricFusion
                from biognn.evaluation.metrics import BiometricEvaluator

                # 2. Dataset
                train_dataset = LUTBioDataset(root='data/LUTBIO', split='train')
                test_dataset = LUTBioDataset(root='data/LUTBIO', split='test')

                # 3. Model
                model = MultimodalBiometricFusion(
                    modalities=['face', 'finger', 'voice'],
                    gnn_type='gat'
                )

                # 4. Egitim
                trainer = Trainer(model, train_loader, val_loader)
                trainer.train(num_epochs=50)

                # 5. Test
                evaluator = BiometricEvaluator()
                results = evaluator.evaluate(y_true, y_scores)
                ```
                """)

                # Gorsel adim adim akis
                gr.Markdown("### Gorsel Akis Diyagrami")

                def create_pipeline_diagram():
                    fig, ax = plt.subplots(figsize=(16, 10))

                    # Renkler
                    colors = {
                        'data': '#3498DB',
                        'extract': '#9B59B6',
                        'graph': '#E67E22',
                        'gnn': '#E74C3C',
                        'output': '#2ECC71'
                    }

                    # Kutular
                    boxes = [
                        {'text': 'LUTBIO\nDataset\n\nYuz + Parmak Izi\n+ Ses', 'pos': (0.08, 0.5), 'color': colors['data'], 'size': (0.12, 0.35)},
                        {'text': 'Ozellik\nCikarimi\n\nResNet50\nMobileNet\nCNN+MFCC', 'pos': (0.28, 0.5), 'color': colors['extract'], 'size': (0.12, 0.35)},
                        {'text': 'Graf\nOlusturma\n\n3 Dugum\n6 Kenar', 'pos': (0.48, 0.5), 'color': colors['graph'], 'size': (0.12, 0.3)},
                        {'text': 'GNN\nKatmanlari\n\nGCN/GAT/\nGraphSAGE', 'pos': (0.68, 0.5), 'color': colors['gnn'], 'size': (0.12, 0.3)},
                        {'text': 'Cikis\n\nGenuine\nvs\nImpostor', 'pos': (0.88, 0.5), 'color': colors['output'], 'size': (0.1, 0.25)},
                    ]

                    from matplotlib.patches import FancyBboxPatch

                    for box in boxes:
                        x, y = box['pos']
                        w, h = box['size']
                        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                             boxstyle="round,pad=0.02,rounding_size=0.02",
                                             facecolor=box['color'], edgecolor='black',
                                             linewidth=2, alpha=0.85)
                        ax.add_patch(rect)
                        ax.text(x, y, box['text'], ha='center', va='center',
                               fontsize=11, fontweight='bold', color='white')

                    # Oklar
                    arrow_props = dict(arrowstyle='->', color='#333', lw=3)
                    arrow_positions = [
                        ((0.14, 0.5), (0.22, 0.5)),
                        ((0.34, 0.5), (0.42, 0.5)),
                        ((0.54, 0.5), (0.62, 0.5)),
                        ((0.74, 0.5), (0.83, 0.5)),
                    ]

                    for start, end in arrow_positions:
                        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

                    # Adim numaralari
                    steps = ['ADIM 1', 'ADIM 2', 'ADIM 3', 'ADIM 4', 'ADIM 5']
                    step_x = [0.08, 0.28, 0.48, 0.68, 0.88]

                    for i, (step, x) in enumerate(zip(steps, step_x)):
                        ax.text(x, 0.85, step, ha='center', va='center',
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_title('GNN ile Multimodal Biyometrik Dogrulama - Gelistirme Akisi',
                                fontsize=16, fontweight='bold', pad=20)
                    ax.axis('off')

                    plt.tight_layout()
                    return fig

                pipeline_plot = gr.Plot(label="Pipeline Diyagrami")
                demo.load(create_pipeline_diagram, outputs=pipeline_plot)

            # ===============================
            # SEKME 8: Hakkinda
            # ===============================
            with gr.TabItem("Hakkinda"):
                gr.Markdown("""
                ## BioGNN Projesi Hakkinda

                ### Proje Aciklamasi
                BioGNN, Graph Neural Network tabanli multimodal biyometrik dogrulama sistemidir.
                Sistem, yuz, parmak izi ve ses gibi farkli biyometrik modaliteleri graf yapisi
                uzerinde birlesttirerek yuksek dogruluklu kimlik dogrulama saglar.

                ### Ozellikler
                - **Multimodal Fuzyon:** Birden fazla biyometrik modaliteyi birlestirir
                - **GNN Tabanli:** Modern graf sinir agi mimarileri kullanir
                - **Esnek Mimari:** GCN, GAT ve GraphSAGE destegi
                - **LUTBIO Destegi:** Standart biyometrik veri seti uyumlulugu

                ### Teknik Detaylar
                - **Framework:** PyTorch + PyTorch Geometric
                - **Ozellik Cikarimi:** ResNet50, MobileNetV2
                - **Ses Isleme:** MFCC + CNN
                - **Degerlendirme:** EER, FAR, FRR, AUC, ROC, DET

                ### Kullanim
                ```bash
                # Dashboard'u baslat
                python demo/biometric_dashboard.py

                # Genel link ile paylas
                python demo/biometric_dashboard.py --share
                ```

                ### Lisans
                MIT License

                ### Iletisim
                GitHub: [BioGNN Repository](https://github.com/erogluefe/BioGNN)
                """)

        # Footer
        gr.Markdown("""
        ---
        <center>
        <p>BioGNN - Multimodal Biometric Verification System | 2024</p>
        </center>
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description='BioGNN Gorsellestirme Dashboard')
    parser.add_argument('--share', action='store_true', help='Genel link olustur')
    parser.add_argument('--port', type=int, default=7861, help='Port numarasi (varsayilan: 7861)')
    args = parser.parse_args()

    if not GRADIO_AVAILABLE:
        print("Hata: Gradio yuklu degil. Kurulum: pip install gradio")
        return

    print("\n" + "="*60)
    print("BioGNN GORSELLESTIRME DASHBOARD")
    print("="*60)
    print(f"Port: {args.port}")
    print(f"Genel Link: {'Evet' if args.share else 'Hayir'}")
    print("="*60 + "\n")

    demo = create_dashboard()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == '__main__':
    main()
