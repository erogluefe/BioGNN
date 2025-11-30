#!/usr/bin/env python3
"""
Dataset Analysis and Visualization Examples

Bu örnek, BioGNN projesindeki dataset görselleştirme araçlarını gösterir.

Örnekler:
1. Multimodal sample grid - Çoklu modalite örnekleri
2. Data distribution dashboard - Dataset istatistikleri
3. Feature space visualization - t-SNE/UMAP embeddings
4. Genuine vs Imposter comparison - Genuine/Imposter karşılaştırması
5. Training monitoring - Eğitim metrikleri
6. Error analysis - Hata analizi
7. Spoofing attack visualization - Sahte giriş saldırıları
8. Augmentation comparison - Veri artırma karşılaştırması

Kullanım:
    python examples/visualize_dataset_analysis.py --example 1
    python examples/visualize_dataset_analysis.py --example 0  # Hepsi
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("DATASET VE ANALYSIS GÖRSELLEŞTİRME ÖRNEKLERİ")
print("=" * 80)


def create_dummy_multimodal_data(num_samples=10):
    """Simüle edilmiş multimodal biometric data oluştur"""
    modalities = {
        'face': torch.randn(num_samples, 3, 112, 112),  # RGB face images
        'fingerprint': torch.randn(num_samples, 1, 96, 96),  # Grayscale fingerprints
        'iris': torch.randn(num_samples, 1, 64, 256),  # Iris images
        'voice': torch.randn(num_samples, 40, 100)  # Voice spectrograms
    }

    subject_ids = np.random.randint(1, 50, size=num_samples)

    quality_scores = {
        'face': np.random.uniform(0.6, 1.0, size=num_samples),
        'fingerprint': np.random.uniform(0.5, 0.95, size=num_samples),
        'iris': np.random.uniform(0.4, 0.9, size=num_samples),
        'voice': np.random.uniform(0.3, 0.85, size=num_samples)
    }

    return modalities, subject_ids, quality_scores


def example_1_multimodal_sample_grid():
    """Örnek 1: Multimodal Sample Grid"""
    print("\n[ÖRNEK 1] Multimodal Sample Grid")

    from biognn.visualization import DatasetVisualizer

    modalities_list = ['face', 'fingerprint', 'iris', 'voice']
    samples, subject_ids, quality_scores = create_dummy_multimodal_data(num_samples=5)

    viz = DatasetVisualizer(modalities_list)

    fig = viz.plot_sample_grid(
        samples=samples,
        num_samples=5,
        subject_ids=subject_ids.tolist(),
        quality_scores=quality_scores,
        title='Multimodal Biometric Samples with Quality Scores',
        save_path='outputs/data_sample_grid.png'
    )

    plt.close()
    print("✓ outputs/data_sample_grid.png kaydedildi")


def example_2_data_distribution_dashboard():
    """Örnek 2: Data Distribution Dashboard"""
    print("\n[ÖRNEK 2] Data Distribution Dashboard")

    from biognn.visualization import plot_data_distribution_dashboard

    # Simüle edilmiş dataset istatistikleri
    dataset_stats = {
        'samples_per_modality': {
            'face': 5000,
            'fingerprint': 4800,
            'iris': 4500,
            'voice': 4200
        },
        'train_val_test_split': {
            'train': 12000,
            'val': 3000,
            'test': 3500
        },
        'class_balance': {
            'genuine': 10000,
            'imposter': 8500
        },
        'samples_per_subject': np.random.poisson(50, 100).tolist(),
        'quality_scores': {
            'face': np.random.beta(8, 2, 5000).tolist(),
            'fingerprint': np.random.beta(7, 3, 4800).tolist(),
            'iris': np.random.beta(6, 4, 4500).tolist(),
            'voice': np.random.beta(5, 3, 4200).tolist()
        }
    }

    fig = plot_data_distribution_dashboard(
        dataset_stats=dataset_stats,
        save_path='outputs/data_distribution_dashboard.png'
    )

    plt.close()
    print("✓ outputs/data_distribution_dashboard.png kaydedildi")


def example_3_feature_space_visualization():
    """Örnek 3: Feature Space Visualization (t-SNE)"""
    print("\n[ÖRNEK 3] Feature Space Visualization")

    from biognn.visualization import plot_feature_space

    # Simüle edilmiş embeddings
    num_subjects = 10
    samples_per_subject = 20
    feature_dim = 512

    embeddings = []
    labels = []

    for subject in range(num_subjects):
        # Her subject için biraz farklı dağılımlı embeddings
        center = np.random.randn(feature_dim) * 5
        subject_embeddings = np.random.randn(samples_per_subject, feature_dim) + center
        embeddings.append(subject_embeddings)
        labels.extend([subject] * samples_per_subject)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    fig = plot_feature_space(
        embeddings=embeddings,
        labels=labels,
        method='tsne',
        title='t-SNE Visualization of Biometric Embeddings',
        save_path='outputs/feature_space_tsne.png',
        show_legend=False
    )

    plt.close()
    print("✓ outputs/feature_space_tsne.png kaydedildi")


def example_4_genuine_vs_imposter():
    """Örnek 4: Genuine vs Imposter Comparison"""
    print("\n[ÖRNEK 4] Genuine vs Imposter Comparison")

    from biognn.visualization import DatasetVisualizer

    modalities_list = ['face', 'fingerprint', 'iris', 'voice']

    genuine_samples, _, _ = create_dummy_multimodal_data(num_samples=3)
    imposter_samples, _, _ = create_dummy_multimodal_data(num_samples=3)

    viz = DatasetVisualizer(modalities_list)

    fig = viz.plot_genuine_vs_imposter(
        genuine_samples=genuine_samples,
        imposter_samples=imposter_samples,
        num_pairs=3,
        save_path='outputs/genuine_vs_imposter.png'
    )

    plt.close()
    print("✓ outputs/genuine_vs_imposter.png kaydedildi")


def example_5_training_monitoring():
    """Örnek 5: Training Monitoring"""
    print("\n[ÖRNEK 5] Training Monitoring Dashboard")

    from biognn.visualization import TrainingMonitor, create_training_dashboard

    modalities_list = ['face', 'fingerprint', 'iris', 'voice']
    monitor = TrainingMonitor(modalities_list)

    # Simüle edilmiş eğitim geçmişi
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        # Realistic training curves
        train_loss = 2.0 * np.exp(-epoch/15) + 0.1 + np.random.normal(0, 0.02)
        val_loss = 2.2 * np.exp(-epoch/15) + 0.15 + np.random.normal(0, 0.03)

        train_acc = 100 * (1 - np.exp(-epoch/10)) * 0.98
        val_acc = 100 * (1 - np.exp(-epoch/10)) * 0.95

        val_eer = 10 * np.exp(-epoch/12) + 0.5
        val_auc = 1 - (0.3 * np.exp(-epoch/15))

        lr = 0.001 * (0.95 ** (epoch // 10))

        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_eer': val_eer,
            'val_auc': val_auc,
            'learning_rate': lr,
            'val_far': val_eer * 0.8,
            'val_frr': val_eer * 1.2
        }

        monitor.log_metrics(metrics, epoch)

    # Dashboard oluştur
    fig = create_training_dashboard(
        monitor=monitor,
        save_path='outputs/training_dashboard.png'
    )

    plt.close()
    print("✓ outputs/training_dashboard.png kaydedildi")


def example_6_error_analysis():
    """Örnek 6: Error Analysis"""
    print("\n[ÖRNEK 6] Error Analysis")

    from biognn.visualization import plot_error_analysis

    # Simüle edilmiş tahminler
    num_samples = 500
    y_true = np.random.randint(0, 2, num_samples)

    # Realistic prediction scores (better separation for genuine/imposter)
    y_scores = np.where(
        y_true == 1,
        np.random.beta(8, 2, num_samples),  # Genuine: high scores
        np.random.beta(2, 8, num_samples)   # Imposter: low scores
    )

    # Predictions based on threshold
    threshold = 0.5
    y_pred = (y_scores > threshold).astype(int)

    sample_ids = [f'sample_{i:04d}' for i in range(num_samples)]

    fig = plot_error_analysis(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        sample_ids=sample_ids,
        top_k=10,
        save_path='outputs/error_analysis.png'
    )

    if fig:
        plt.close()
        print("✓ outputs/error_analysis.png kaydedildi")


def example_7_spoofing_attack_visualization():
    """Örnek 7: Spoofing Attack Visualization"""
    print("\n[ÖRNEK 7] Spoofing Attack Visualization")

    from biognn.visualization import plot_spoofing_attack_comparison

    modalities_list = ['face', 'fingerprint', 'iris']

    genuine_samples, _, _ = create_dummy_multimodal_data(num_samples=1)
    spoofed_samples = {
        'face': torch.randn(3, 3, 112, 112),  # 3 attack types
        'fingerprint': torch.randn(3, 1, 96, 96),
        'iris': torch.randn(3, 1, 64, 256)
    }

    attack_types = ['Print Attack', 'Replay Attack', 'Mask Attack']
    detection_scores = np.random.uniform(0.7, 0.95, 3)  # Detection confidence

    fig = plot_spoofing_attack_comparison(
        genuine_samples=genuine_samples,
        spoofed_samples=spoofed_samples,
        attack_types=attack_types,
        detection_scores=detection_scores,
        modalities=modalities_list,
        save_path='outputs/spoofing_attack_comparison.png'
    )

    plt.close()
    print("✓ outputs/spoofing_attack_comparison.png kaydedildi")


def example_8_augmentation_comparison():
    """Örnek 8: Augmentation Comparison"""
    print("\n[ÖRNEK 8] Data Augmentation Comparison")

    from biognn.visualization import plot_augmentation_comparison

    modalities_list = ['face', 'fingerprint']

    original_samples, _, _ = create_dummy_multimodal_data(num_samples=1)

    # Farklı augmentation türleri (simüle edilmiş)
    augmented_samples_list = [
        {
            'face': torch.randn(1, 3, 112, 112) * 1.1,  # Brightness
            'fingerprint': torch.randn(1, 1, 96, 96) * 0.9
        },
        {
            'face': torch.randn(1, 3, 112, 112) + 0.1,  # Contrast
            'fingerprint': torch.randn(1, 1, 96, 96) - 0.05
        },
        {
            'face': torch.randn(1, 3, 112, 112) * 0.95,  # Rotation
            'fingerprint': torch.randn(1, 1, 96, 96) * 1.05
        }
    ]

    augmentation_names = ['Brightness Adjustment', 'Contrast Enhancement', 'Random Rotation']

    fig = plot_augmentation_comparison(
        original_samples={'face': original_samples['face'][:1],
                         'fingerprint': original_samples['fingerprint'][:1]},
        augmented_samples_list=augmented_samples_list,
        augmentation_names=augmentation_names,
        modalities=modalities_list,
        save_path='outputs/augmentation_comparison.png'
    )

    plt.close()
    print("✓ outputs/augmentation_comparison.png kaydedildi")


def main():
    parser = argparse.ArgumentParser(description='Dataset Analysis Visualization Examples')
    parser.add_argument(
        '--example',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
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

    examples = {
        1: example_1_multimodal_sample_grid,
        2: example_2_data_distribution_dashboard,
        3: example_3_feature_space_visualization,
        4: example_4_genuine_vs_imposter,
        5: example_5_training_monitoring,
        6: example_6_error_analysis,
        7: example_7_spoofing_attack_visualization,
        8: example_8_augmentation_comparison
    }

    if args.example == 0:
        # Tüm örnekleri çalıştır
        print("\n" + "="*80)
        print("TÜM ÖRNEKLER ÇALIŞTIRILIYOR")
        print("="*80)
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"✗ Hata: {e}")
            print("-"*80)
    else:
        # Belirli bir örneği çalıştır
        try:
            examples[args.example]()
        except Exception as e:
            print(f"✗ Hata: {e}")

    print("\n" + "="*80)
    print("GÖRSELLEŞTİRME TAMAMLANDI!")
    print("="*80)
    print(f"\nTüm grafikler '{args.output}/' klasörüne kaydedildi.")
    print("\nDikkat: Bu örnekler simüle edilmiş verilerle çalışır.")
    print("Gerçek veri ile kullanmak için kendi dataset'inizi yükleyin.")


if __name__ == '__main__':
    main()
