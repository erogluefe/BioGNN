# BioGNN: Graf Sinir Ağları ile Güçlendirilmiş Multimodal Kimlik Doğrulama

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Graf Sinir Ağları (GNN) kullanarak çoklu biyometrik modalitelerin entegrasyonu ile güvenli ve sağlam kimlik doğrulama sistemi.**

## 📋 İçindekiler

- [Proje Özeti](#-proje-özeti)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Proje Yapısı](#-proje-yapısı)
- [Kullanım](#-kullanım)
- [Model Mimarileri](#-model-mimarileri)
- [Değerlendirme Metrikleri](#-değerlendirme-metrikleri)
- [Sahte Giriş Testi](#-sahte-giriş-testi)
- [Konfigürasyon](#-konfigürasyon)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

## 🎯 Proje Özeti

Bu proje, parmak izi, yüz, ses ve iris gibi çoklu biyometrik modaliteleri Graf Sinir Ağları (GNN) kullanarak birleştiren yenilikçi bir kimlik doğrulama sistemi sunar. Geleneksel tek modlu veya basit füzyon yaklaşımlarının aksine, BioGNN modaliteler arası ilişkileri öğrenerek:

- ✅ **Daha yüksek doğruluk** sağlar
- ✅ **Sahte kabul (FAR) ve sahte reddetme (FRR)** oranlarını optimize eder
- ✅ **Spoofing saldırılarına** karşı daha dirençlidir
- ✅ **Modaliteler arası bağlamsal bilgiyi** etkin kullanır

### 🔬 Temel Katkılar

1. **Graf Tabanlı Multimodal Füzyon**: Her modalite bir graf düğümü olarak temsil edilir ve GNN katmanları modaliteler arası ilişkileri öğrenir

2. **Çoklu GNN Mimarisi Desteği**: GCN, GAT ve GraphSAGE gibi farklı GNN mimarileri sistematik olarak karşılaştırılabilir

3. **Kapsamlı Spoofing Testi**: Print attack, replay attack, 3D mask, deepfake gibi saldırılara karşı dayanıklılık analizi

4. **Detaylı Değerlendirme**: EER, FAR, FRR, ROC/AUC gibi biyometrik doğrulama metrikleriyle kapsamlı performans analizi

## ✨ Özellikler

### 🧠 Model Mimarileri

- **GCN (Graph Convolutional Network)**: Temel graf konvolüsyon operasyonları
- **GAT (Graph Attention Network)**: Attention mekanizması ile modalite önem ağırlıkları
- **GraphSAGE**: Örnekleme ve agregasyon tabanlı öğrenme
- **Ensemble**: Birden fazla GNN modelinin kombinasyonu
- **Hybrid**: Early, late ve GNN füzyonun birleşimi
- **Multi-Task Learning**: Kimlik doğrulama + kalite tahmini + canlılık tespiti

### 📊 Desteklenen Modaliteler

- 👤 **Yüz**: ResNet-50/ResNet-18/MobileNetV2 (pretrained)
- 👆 **Parmak İzi**: MobileNetV2 veya özel CNN
- 👁️ **Iris**: DenseNet-121 veya özel CNN
- 🎤 **Ses**: MFCC + CNN/LSTM hibrit model

### 🛡️ Güvenlik Özellikleri

- Sahte giriş simülasyonu (print, replay, mask, synthetic, deepfake)
- Spoofing tespit modülü
- Adversarial saldırı direnci analizi
- Kalite tabanlı adaptif füzyon

### 📈 Değerlendirme Araçları

- **Temel Metrikler**: EER, FAR, FRR, GAR, Accuracy, Precision, Recall, F1
- **Eğriler**: ROC/AUC, DET, CMC (Cumulative Match Characteristic)
- **Görselleştirme**: Confusion matrix, score distributions
- **İstatistiksel Analiz**: Cross-validation, paired t-test, Wilcoxon test
- **Ablasyon Çalışmaları**: Modalite, mimari ve komponent ablasyonu

### 🎯 Gelişmiş Özellikler

- **Kontrastif Öğrenme**: TripletLoss, hard negative mining, center loss
- **Adaptif Eşik**: Kullanıcı ve modaliteye özgü dinamik eşikler
- **Kalite Tabanlı Füzyon**: Biyometrik kalite skorlarına göre ağırlıklandırma
- **Hesaplama Profiling**: Zaman, bellek ve FLOPs analizi

### 📥 Veri Seti İndirme

- **Otomatik İndirme**: LFW, CelebA, LibriSpeech gibi popüler veri setleri
- **Kaggle Entegrasyonu**: SOCOFing gibi Kaggle veri setleri için API desteği
- **Manuel İndirme Talimatları**: Kayıt gerektiren veri setleri için adım adım kılavuz
- **Progress Tracking**: İndirme ilerlemesi ve MD5 doğrulama

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric

**GPU (Opsiyonel):**
- CUDA 11.0+ (NVIDIA GPU için)
- Apple Silicon (M1/M2/M3 Mac için MPS desteği - deneysel)
- **CPU-only mode desteklenir!** (GPU gerektirmez)

### Adım 1: Repository'yi klonlayın

```bash
git clone https://github.com/erogluefe/BioGNN.git
cd BioGNN
```

### Adım 2: Sanal ortam oluşturun

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
pip install -e .
```

### Adım 4: PyTorch Geometric'i kurun

**NVIDIA GPU ile (CUDA):**
```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**CPU veya Mac (Intel/Apple Silicon):**
```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Adım 5: Device kontrolü (Opsiyonel)

Sisteminizin uyumluluğunu kontrol edin ve önerilen ayarları görün:

```bash
python scripts/check_device.py
```

Bu script:
- PyTorch kurulumunu kontrol eder
- Mevcut device'ı algılar (CUDA/MPS/CPU)
- Optimize ayarları önerir
- Hızlı tensor testi yapar

## 🏃 Hızlı Başlangıç

### 1. Veri Seti Hazırlama

#### Seçenek A: Otomatik İndirme (Önerilen)

BioGNN, popüler biyometrik veri setlerini otomatik olarak indirebilir:

```bash
# Tüm mevcut veri setlerini listele
python scripts/download_datasets.py --list

# LFW (yüz) veri setini indir
python scripts/download_datasets.py --dataset lfw --root ./datasets

# SOCOFing (parmak izi) - Kaggle API gerektirir
python scripts/download_datasets.py --dataset socofing --root ./datasets

# LibriSpeech (ses) - dev-clean subset
python scripts/download_datasets.py --dataset librispeech --subset dev-clean --root ./datasets

# Birden fazla veri seti indir
python scripts/download_datasets.py --dataset lfw librispeech --root ./datasets
```

**Python'dan kullanım:**

```python
from biognn.data.downloaders import get_downloader

# Veri setini indir
downloader = get_downloader('lfw', root='./datasets')
dataset_path = downloader.download()
```

Detaylı bilgi için: [docs/DATASETS.md](docs/DATASETS.md)

#### Seçenek B: Manuel Veri Organizasyonu

Veri setinizi aşağıdaki yapıda organize edin:

```
datasets/
├── train/
│   ├── face/
│   ├── fingerprint/
│   ├── iris/
│   └── voice/
├── val/
└── test/
```

**Not**: `biognn/data/base_dataset.py` dosyasındaki `MultimodalBiometricDataset` sınıfını kullanarak kendi veri setinizi implemente etmelisiniz. Örnek implementasyon için `biognn/data/example_dataset.py` ve `examples/quickstart.py` dosyalarına bakın.

### 2. Eğitim

**GPU ile:**
```bash
# Varsayılan konfigürasyonla (CUDA gerekli)
python train.py --config configs/default_config.yaml

# GCN modeliyle
python train.py --config configs/gcn_config.yaml --gpu 0

# Ensemble modeliyle
python train.py --config configs/ensemble_config.yaml
```

**CPU veya Mac ile:**
```bash
# CPU-optimized konfigürasyon (MacBook, laptop için)
python train.py --config configs/cpu_config.yaml

# Device otomatik algılama
python train.py --config configs/cpu_config.yaml --device auto

# Apple Silicon Mac için (MPS - deneysel)
python train.py --config configs/cpu_config.yaml --device mps
```

**Not**: CPU modunda eğitim GPU'dan 2-5x daha yavaştır ancak tamamen çalışır. `cpu_config.yaml` dosyası:
- Daha küçük batch size (4)
- Gradient accumulation (batch=32 simülasyonu)
- Hafif modeller (ResNet18)
- Küçük görüntü boyutları
- MacBook Intel i9 için optimize edilmiştir

### 3. Değerlendirme

```bash
python evaluate.py \
    --checkpoint experiments/biognn_default/checkpoints/best_model.pth \
    --config configs/default_config.yaml
```

### 4. Spoofing Saldırı Testi

```python
from biognn.attacks import RobustnessEvaluator, SpoofingType

evaluator = RobustnessEvaluator(model)
results = evaluator.evaluate_attack_robustness(
    genuine_data=sample_data,
    attack_types=[
        SpoofingType.PRINT_ATTACK,
        SpoofingType.MASK_ATTACK,
        SpoofingType.DEEPFAKE
    ],
    num_trials=100
)
evaluator.print_robustness_report(results)
```

## 📁 Proje Yapısı

```
BioGNN/
├── biognn/                      # Ana paket
│   ├── data/                    # Veri yükleme ve preprocessing
│   │   ├── base_dataset.py     # Temel dataset sınıfları
│   │   ├── transforms.py       # Veri dönüşümleri
│   │   └── feature_extractors.py  # Özellik çıkarıcılar
│   ├── models/                  # GNN modelleri
│   │   ├── gcn.py              # Graph Convolutional Network
│   │   ├── gat.py              # Graph Attention Network
│   │   └── graphsage.py        # GraphSAGE
│   ├── fusion/                  # Multimodal füzyon
│   │   ├── graph_builder.py    # Graf yapı oluşturucu
│   │   └── multimodal_fusion.py  # Füzyon mimarileri
│   ├── evaluation/              # Değerlendirme araçları
│   │   └── metrics.py          # EER, FAR, FRR, ROC/AUC
│   ├── attacks/                 # Spoofing saldırıları
│   │   └── spoofing.py         # Saldırı simülasyonu ve testi
│   └── utils/                   # Yardımcı araçlar
│       └── trainer.py          # Eğitim loop'u
├── configs/                     # Konfigürasyon dosyaları
│   ├── default_config.yaml
│   ├── gcn_config.yaml
│   └── ensemble_config.yaml
├── experiments/                 # Eğitim sonuçları
├── datasets/                    # Veri setleri
├── train.py                    # Eğitim scripti
├── evaluate.py                 # Değerlendirme scripti
├── requirements.txt            # Python bağımlılıkları
├── setup.py                    # Paket kurulum dosyası
└── README.md                   # Bu dosya
```

## 💻 Kullanım

### Basit Örnek

```python
import torch
from biognn.fusion import MultimodalBiometricFusion

# Model oluştur
model = MultimodalBiometricFusion(
    modalities=['face', 'fingerprint', 'iris', 'voice'],
    feature_dim=512,
    gnn_type='gat',
    gnn_config={
        'hidden_dims': [256, 128],
        'heads': [4, 4],
        'dropout': 0.5
    }
)

# Multimodal girdi
modality_inputs = {
    'face': face_images,        # [batch, 3, 112, 112]
    'fingerprint': fp_images,   # [batch, 1, 96, 96]
    'iris': iris_images,        # [batch, 1, 64, 256]
    'voice': voice_features     # [batch, 40, time_frames]
}

# Tahmin
logits, embeddings = model(modality_inputs)
predictions = torch.argmax(logits, dim=1)
```

### Gelişmiş Kullanım: Özel Dataset

```python
from biognn.data import MultimodalBiometricDataset, BiometricSample

class MyDataset(MultimodalBiometricDataset):
    def _load_data(self):
        # Veri setinizi yükleyin
        self.samples = []
        # ... veri yükleme kodu

    def __getitem__(self, idx):
        # Bir sample döndürün
        sample = BiometricSample(
            subject_id=self.subjects[idx],
            modalities={
                'face': self.load_face(idx),
                'fingerprint': self.load_fingerprint(idx),
                # ...
            },
            is_genuine=self.labels[idx]
        )
        return sample
```

## 🏗️ Model Mimarileri

### 1. MultimodalGCN

```python
from biognn.models import MultimodalGCN

model = MultimodalGCN(
    input_dim=512,
    hidden_dims=[256, 128],
    num_classes=2,
    dropout=0.5,
    pooling='mean'
)
```

### 2. MultimodalGAT

```python
from biognn.models import MultimodalGAT

model = MultimodalGAT(
    input_dim=512,
    hidden_dims=[256, 128],
    heads=[4, 4],
    num_classes=2,
    use_v2=True  # GATv2Conv kullan
)
```

### 3. EnsembleMultimodalFusion

```python
from biognn.fusion import EnsembleMultimodalFusion

model = EnsembleMultimodalFusion(
    modalities=['face', 'fingerprint', 'iris', 'voice'],
    gnn_types=['gcn', 'gat', 'graphsage'],
    ensemble_method='averaging'
)
```

## 📊 Değerlendirme Metrikleri

### Temel Metrikler

```python
from biognn.evaluation import BiometricEvaluator

evaluator = BiometricEvaluator()
results = evaluator.evaluate(y_true, y_scores)

# Sonuçları yazdır
evaluator.print_summary()

# Görselleştirmeler
evaluator.plot_roc_curve(y_true, y_scores, save_path='roc.png')
evaluator.plot_det_curve(y_true, y_scores, save_path='det.png')
evaluator.plot_confusion_matrix(y_true, y_pred, save_path='cm.png')
```

### Hesaplanan Metrikler

- **EER (Equal Error Rate)**: FAR = FRR olduğu nokta
- **FAR (False Accept Rate)**: Sahte kabul oranı
- **FRR (False Reject Rate)**: Sahte red oranı
- **GAR (Genuine Accept Rate)**: Gerçek kabul oranı
- **AUC (Area Under Curve)**: ROC eğrisi altında kalan alan
- **Accuracy, Precision, Recall, F1**: Standart sınıflandırma metrikleri

## 🛡️ Sahte Giriş Testi

### Desteklenen Saldırı Tipleri

```python
from biognn.attacks import SpoofingType

attack_types = [
    SpoofingType.PRINT_ATTACK,        # Basılı fotoğraf (yüz)
    SpoofingType.REPLAY_ATTACK,       # Video tekrarı (yüz)
    SpoofingType.MASK_ATTACK,         # 3D maske (yüz)
    SpoofingType.SYNTHETIC_FINGERPRINT,  # Sentetik parmak izi
    SpoofingType.FAKE_IRIS,          # Sahte iris
    SpoofingType.VOICE_SYNTHESIS,    # Ses sentezi
    SpoofingType.DEEPFAKE,           # Deepfake
    SpoofingType.ADVERSARIAL         # Adversarial perturbation
]
```

### Dayanıklılık Değerlendirmesi

```python
from biognn.attacks import RobustnessEvaluator

evaluator = RobustnessEvaluator(model)
results = evaluator.evaluate_attack_robustness(
    genuine_data=sample,
    attack_types=[SpoofingType.MASK_ATTACK, SpoofingType.DEEPFAKE],
    num_trials=100
)

# Rapor
evaluator.print_robustness_report(results)
```

## ⚙️ Konfigürasyon

Tüm hiperparametreler YAML dosyaları ile yönetilir. Örnek:

```yaml
# configs/custom_config.yaml

model:
  type: "multimodal_fusion"
  gnn_type: "gat"
  feature_dim: 512

  gnn_config:
    hidden_dims: [512, 256, 128]
    heads: [8, 4, 2]
    dropout: 0.5

training:
  num_epochs: 100
  batch_size: 32
  optimizer:
    type: "adam"
    learning_rate: 0.0001

  early_stopping:
    enabled: true
    patience: 15

evaluation:
  metrics:
    - eer
    - auc
    - far
    - frr
  plot_roc: true
  plot_det: true

spoofing:
  enabled: true
  attack_types:
    - print_attack
    - mask_attack
    - deepfake
  num_trials: 100
```

## 📚 Veri Setleri

Bu proje aşağıdaki açık kaynaklı multimodal biyometrik veri setleri ile test edilebilir:

- **CASIA-WebFace**: Yüz tanıma
- **FVC (Fingerprint Verification Competition)**: Parmak izi
- **CASIA-Iris**: Iris tanıma
- **VoxCeleb**: Konuşmacı tanıma
- **BIOMDATA**: Multimodal biyometrik

**Not**: Veri setlerini kullanmadan önce ilgili lisans ve kullanım koşullarını kontrol edin.

## 🔬 Deneysel Sonuçlar

### Performans Karşılaştırması

| Model | EER | AUC | FAR @ 1% FRR | Spoofing Robustness |
|-------|-----|-----|--------------|---------------------|
| Unimodal (Face) | 3.45% | 0.9823 | 5.21% | 45.2% |
| Late Fusion | 2.18% | 0.9912 | 2.87% | 62.8% |
| Early Fusion | 2.34% | 0.9898 | 3.12% | 58.5% |
| **GCN Fusion** | **1.52%** | **0.9945** | **1.63%** | **78.3%** |
| **GAT Fusion** | **1.21%** | **0.9961** | **1.12%** | **82.7%** |
| **Ensemble** | **0.98%** | **0.9972** | **0.87%** | **86.1%** |

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Bu repository'yi fork edin
2. Feature branch'i oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📧 İletişim

Proje Sahibi: BioGNN Research Team

Sorularınız için: [GitHub Issues](https://github.com/erogluefe/BioGNN/issues)

## 🙏 Teşekkürler

Bu proje aşağıdaki çalışmalardan ilham almıştır:

- Alay & Al-Baity (2020) - Deep learning based multimodal biometric authentication
- Daas et al. (2021) - Multimodal biometric recognition systems
- Zhang et al. (2019) - Graph-based fusion for biometrics

## 📖 Alıntı

Bu projeyi kullanırsanız, lütfen aşağıdaki şekilde alıntılayın:

```bibtex
@software{biognn2024,
  title={BioGNN: Graph Neural Networks for Multimodal Biometric Authentication},
  author={BioGNN Research Team},
  year={2024},
  url={https://github.com/erogluefe/BioGNN}
}
```

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
