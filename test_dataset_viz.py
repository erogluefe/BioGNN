#!/usr/bin/env python3
"""
Dataset visualization test - dependency bypass
"""

import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util

print("=" * 80)
print("DATASET GÖRSELLEŞTIRME TESTİ")
print("=" * 80)

# Manuel modül yükleme (dependency zincirini atlayarak)
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Modülleri yükle
base_path = Path("/home/user/BioGNN/biognn/visualization")
data_viz = load_module("data_viz", base_path / "data_viz.py")
training_viz = load_module("training_viz", base_path / "training_viz.py")
analysis_viz = load_module("analysis_viz", base_path / "analysis_viz.py")

# Test 1: Dataset stats dashboard
print("\n[TEST 1] Data Distribution Dashboard")
dataset_stats = {
    'samples_per_modality': {'face': 5000, 'fingerprint': 4800, 'iris': 4500, 'voice': 4200},
    'train_val_test_split': {'train': 12000, 'val': 3000, 'test': 3500},
    'class_balance': {'genuine': 10000, 'imposter': 8500},
    'samples_per_subject': np.random.poisson(50, 100).tolist(),
    'quality_scores': {
        'face': np.random.beta(8, 2, 5000).tolist(),
        'fingerprint': np.random.beta(7, 3, 4800).tolist()
    }
}

fig = data_viz.plot_data_distribution_dashboard(dataset_stats, save_path='outputs/test_data_distribution.png')
plt.close()
print("✓ outputs/test_data_distribution.png")

# Test 2: Feature space visualization
print("\n[TEST 2] Feature Space (t-SNE)")
num_subjects = 10
samples_per_subject = 20
feature_dim = 128  # Reduced for faster t-SNE

embeddings = []
labels = []

for subject in range(num_subjects):
    center = np.random.randn(feature_dim) * 5
    subject_embeddings = np.random.randn(samples_per_subject, feature_dim) + center
    embeddings.append(subject_embeddings)
    labels.extend([subject] * samples_per_subject)

embeddings = np.vstack(embeddings)
labels = np.array(labels)

fig = data_viz.plot_feature_space(
    embeddings=embeddings,
    labels=labels,
    method='pca',  # PCA is faster than t-SNE for testing
    title='PCA Visualization of Embeddings',
    save_path='outputs/test_feature_space.png'
)
plt.close()
print("✓ outputs/test_feature_space.png")

# Test 3: Training monitoring
print("\n[TEST 3] Training Monitor")
modalities = ['face', 'fingerprint', 'iris']
monitor = training_viz.TrainingMonitor(modalities)

for epoch in range(1, 30):
    metrics = {
        'train_loss': 2.0 * np.exp(-epoch/10) + 0.1,
        'val_loss': 2.2 * np.exp(-epoch/10) + 0.15,
        'train_accuracy': 100 * (1 - np.exp(-epoch/8)) * 0.98,
        'val_accuracy': 100 * (1 - np.exp(-epoch/8)) * 0.95,
        'learning_rate': 0.001 * (0.95 ** (epoch // 10))
    }
    monitor.log_metrics(metrics, epoch)

fig = monitor.plot_training_curves(save_path='outputs/test_training_curves.png')
plt.close()
print("✓ outputs/test_training_curves.png")

# Test 4: Error analysis
print("\n[TEST 4] Error Analysis")
num_samples = 300
y_true = np.random.randint(0, 2, num_samples)
y_scores = np.where(
    y_true == 1,
    np.random.beta(8, 2, num_samples),
    np.random.beta(2, 8, num_samples)
)
y_pred = (y_scores > 0.5).astype(int)

fig = analysis_viz.plot_error_analysis(
    y_true=y_true,
    y_pred=y_pred,
    y_scores=y_scores,
    top_k=10,
    save_path='outputs/test_error_analysis.png'
)
plt.close()
print("✓ outputs/test_error_analysis.png")

# Test 5: Quality impact analysis
print("\n[TEST 5] Quality Impact Analysis")
quality_scores = np.random.beta(5, 2, 200)
# Higher quality -> higher accuracy
accuracies = (quality_scores > 0.5).astype(int)
# Add some noise
noise = np.random.rand(200) < (quality_scores * 0.5)
accuracies = (accuracies | noise).astype(int)

fig = analysis_viz.plot_quality_impact_analysis(
    quality_scores=quality_scores,
    accuracies=accuracies,
    modality_name='Face',
    save_path='outputs/test_quality_impact.png'
)
plt.close()
print("✓ outputs/test_quality_impact.png")

print("\n" + "=" * 80)
print("TÜM TESTLER BAŞARILI!")
print("=" * 80)
print("\nOluşturulan dosyalar:")
print("  - outputs/test_data_distribution.png")
print("  - outputs/test_feature_space.png")
print("  - outputs/test_training_curves.png")
print("  - outputs/test_error_analysis.png")
print("  - outputs/test_quality_impact.png")
