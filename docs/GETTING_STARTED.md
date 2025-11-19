# BioGNN - Kullanım Kılavuzu

Bu dokümantasyon, BioGNN projesini kullanmaya başlamak için gereken tüm adımları içerir.

## 📋 İçindekiler

1. [Gereksinimler](#gereksinimler)
2. [Kurulum](#kurulum)
3. [Veri Seti Hazırlama](#veri-seti-hazırlama)
4. [Hızlı Başlangıç](#hızlı-başlangıç)
5. [Kendi Veri Setinizi Kullanma](#kendi-veri-setinizi-kullanma)
6. [Model Eğitimi](#model-eğitimi)
7. [Değerlendirme](#değerlendirme)
8. [Sık Sorulan Sorular](#sık-sorulan-sorular)

## ⚙️ Gereksinimler

### Yazılım Gereksinimleri

- Python 3.8 veya üzeri
- CUDA 11.0+ (GPU kullanımı için - opsiyonel ama önerilir)
- 8GB+ RAM (16GB önerilir)
- 10GB+ disk alanı

### Donanım Önerileri

**Minimum**:
- CPU: 4 core
- RAM: 8GB
- GPU: NVIDIA GPU (4GB+ VRAM)

**Önerilen**:
- CPU: 8+ core
- RAM: 16GB+
- GPU: NVIDIA RTX 3090 / A100 (12GB+ VRAM)

## 🚀 Kurulum

### 1. Repository'yi Klonlayın

```bash
git clone https://github.com/erogluefe/BioGNN.git
cd BioGNN
```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 3. Bağımlılıkları Yükleyin

```bash
# Temel bağımlılıklar
pip install -r requirements.txt

# Paketi editable modda yükleyin
pip install -e .

# PyTorch Geometric (CUDA 11.8 için)
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CPU-only kullanıyorsanız:
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### 4. Kurulumu Doğrulayın

```bash
python -c "import biognn; print('BioGNN version:', biognn.__version__)"
python -c "import torch_geometric; print('PyG installed successfully')"
```

## 📁 Veri Seti Hazırlama

### Seçenek 1: Sentetik Veri (Test İçin)

Test ve geliştirme için sentetik veri kullanabilirsiniz:

```bash
# Hızlı başlangıç örneğini çalıştırın
python examples/quickstart.py
```

Bu, gerçek veri olmadan sistemi test etmenizi sağlar.

### Seçenek 2: Gerçek Veri Seti Kullanma

#### Önerilen Açık Veri Setleri

1. **Yüz Tanıma**:
   - [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) (10,575 kişi)
   - [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
   - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

2. **Parmak İzi**:
   - [FVC2004](http://bias.csr.unibo.it/fvc2004/) (Fingerprint Verification Competition)
   - [SOCOFing](https://www.kaggle.com/datasets/ruizgara/socofing)

3. **Iris**:
   - [CASIA-Iris-V4](http://biometrics.idealtest.org/)
   - [UBIRIS](http://iris.di.ubi.pt/)

4. **Ses**:
   - [VoxCeleb1/2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
   - [LibriSpeech](http://www.openslr.org/12)

#### Veri Organizasyonu

Veri setinizi şu yapıda organize edin:

```
datasets/
├── train/
│   ├── subject_001/
│   │   ├── face_001.jpg
│   │   ├── face_002.jpg
│   │   ├── fingerprint_001.png
│   │   ├── iris_001.png
│   │   └── voice_001.wav
│   ├── subject_002/
│   │   └── ...
│   └── ...
├── val/
│   └── ... (aynı yapı)
└── test/
    └── ... (aynı yapı)
```

**Not**: Her modalite için dosya isimleri `{modality}_{index}.{ext}` formatında olmalı.

## 🏃 Hızlı Başlangıç

### Sentetik Veri ile Test

```bash
# Hızlı başlangıç scripti
python examples/quickstart.py
```

Bu script:
- ✅ Sentetik veri oluşturur
- ✅ Model eğitir (3 epoch)
- ✅ Değerlendirme yapar
- ✅ ROC ve DET eğrileri çizer

### Gerçek Veri ile Eğitim

```bash
# Varsayılan konfigürasyonla
python train.py --config configs/default_config.yaml

# GAT modeliyle
python train.py --config configs/default_config.yaml --gpu 0
```

## 🔧 Kendi Veri Setinizi Kullanma

### 1. Dataset Sınıfı Oluşturun

`biognn/data/example_dataset.py` dosyasını template olarak kullanın:

```python
from biognn.data import MultimodalBiometricDataset, BiometricSample
from biognn.data.example_dataset import ExampleMultimodalDataset

# Kendi dataset sınıfınızı oluşturun
class MyDataset(ExampleMultimodalDataset):
    def __init__(self, root, modalities, split='train', transform=None):
        super().__init__(root, modalities, split, transform, download=False)

    def _load_data(self):
        # Kendi veri yükleme mantığınızı buraya yazın
        pass
```

### 2. Dataset'i Kullanın

```python
from biognn.data import get_default_transforms

# Transforms oluştur
transforms = {
    mod: get_default_transforms(mod, augment=True)
    for mod in ['face', 'fingerprint', 'iris', 'voice']
}

# Dataset oluştur
dataset = MyDataset(
    root='./datasets',
    modalities=['face', 'fingerprint', 'iris', 'voice'],
    split='train',
    transform=transforms
)

# DataLoader oluştur
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 🎓 Model Eğitimi

### Basit Eğitim

```python
from biognn.fusion import MultimodalBiometricFusion
from biognn.utils import Trainer

# Model oluştur
model = MultimodalBiometricFusion(
    modalities=['face', 'fingerprint', 'iris', 'voice'],
    feature_dim=512,
    gnn_type='gat',
    gnn_config={'hidden_dims': [256, 128], 'heads': [4, 4]}
)

# Trainer oluştur
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda'
)

# Eğit
trainer.train(num_epochs=100, save_best=True)
```

### Gelişmiş Eğitim (Contrastive Learning)

```python
from biognn.utils import CombinedLoss

# Combined loss kullan
criterion = CombinedLoss(
    num_classes=1000,
    feature_dim=512,
    use_triplet=True,
    use_center=True
)

# Trainer'da kullan
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    device='cuda'
)
```

### Multi-Task Learning

```python
from biognn.models import MultiTaskBiometricModel

model = MultiTaskBiometricModel(
    modalities=['face', 'fingerprint', 'iris', 'voice'],
    use_quality_task=True,
    use_liveness_task=True
)

# Eğitim sırasında 3 loss hesaplanır:
# 1. Verification loss
# 2. Quality estimation loss
# 3. Liveness detection loss
```

## 📊 Değerlendirme

### Temel Değerlendirme

```python
from biognn.evaluation import BiometricEvaluator

evaluator = BiometricEvaluator()
results = evaluator.evaluate(y_true, y_scores)
evaluator.print_summary()

# Görselleştirmeler
evaluator.plot_roc_curve(y_true, y_scores, save_path='roc.png')
evaluator.plot_det_curve(y_true, y_scores, save_path='det.png')
evaluator.plot_confusion_matrix(y_true, y_pred, save_path='cm.png')
```

### CMC Eğrisi (Identification)

```python
from biognn.evaluation import CMCEvaluator

evaluator = CMCEvaluator(max_rank=20)
results = evaluator.evaluate(
    query_features,
    gallery_features,
    query_labels,
    gallery_labels
)

evaluator.print_summary()  # Rank-1, Rank-5, Rank-10
evaluator.plot(save_path='cmc.png')
```

### Ablasyon Çalışması

```bash
# Modalite ablasyonu
python scripts/ablation_study.py \
    --study modality \
    --checkpoint best_model.pth \
    --output results/ablation

# Mimari ablasyonu
python scripts/ablation_study.py \
    --study architecture \
    --checkpoint best_model.pth
```

### İstatistiksel Analiz

```bash
# Cross-validation
python scripts/statistical_analysis.py \
    --analysis cv \
    --n_folds 5

# Leave-one-subject-out
python scripts/statistical_analysis.py \
    --analysis loso
```

## ❓ Sık Sorulan Sorular

### Q: Veri setim sadece 2 modalite içeriyor, kullanabilir miyim?

**A**: Evet! Model herhangi bir modalite kombinasyonuyla çalışır:

```python
model = MultimodalBiometricFusion(
    modalities=['face', 'fingerprint'],  # Sadece 2 modalite
    feature_dim=512,
    gnn_type='gat'
)
```

### Q: GPU olmadan çalıştırabilir miyim?

**A**: Evet, ama yavaş olacaktır. CPU kullanımı için:

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    device='cpu',
    use_amp=False  # AMP sadece GPU'da çalışır
)
```

### Q: Kendi feature extractor'ımı kullanabilir miyim?

**A**: Evet:

```python
from biognn.data.feature_extractors import FaceFeatureExtractor

# Kendi extractor'ınız
class MyFaceExtractor(FaceFeatureExtractor):
    def __init__(self, feature_dim=512):
        super().__init__(
            backbone='resnet50',
            pretrained=True,
            feature_dim=feature_dim
        )

    # forward() metodunu override edebilirsiniz
```

### Q: Eğitim çok yavaş, nasıl hızlandırabilirim?

**A**: Birkaç öneri:

1. **Batch size'ı artırın**: `batch_size=64` (GPU belleği yetiyorsa)
2. **AMP kullanın**: `use_amp=True`
3. **Num workers artırın**: `DataLoader(..., num_workers=4)`
4. **Daha küçük model**: `hidden_dims=[128, 64]`
5. **Backbone freeze**: `freeze_backbone=True`

### Q: Out of memory hatası alıyorum

**A**: Çözümler:

1. Batch size'ı azaltın: `batch_size=8`
2. Feature dimension azaltın: `feature_dim=256`
3. Gradient accumulation kullanın
4. Mixed precision training: `use_amp=True`

### Q: Pretrained modeller var mı?

**A**: Şu anda hayır. Ancak feature extractor'lar (ResNet, MobileNet, DenseNet) ImageNet pretrained weights kullanır:

```python
model = FaceFeatureExtractor(
    backbone='resnet50',
    pretrained=True  # ImageNet weights
)
```

### Q: Veri setim dengesiz (çok fazla genuine, az impostor)

**A**: `VerificationPairDataset`'te ratio'yu ayarlayın:

```python
dataset = VerificationPairDataset(
    base_dataset=base,
    num_pairs=10000,
    genuine_ratio=0.3  # %30 genuine, %70 impostor
)
```

## 📚 Ek Kaynaklar

- **Detaylı API Dokümantasyonu**: Kod içindeki docstring'leri okuyun
- **Örnek Konfigürasyonlar**: `configs/` klasörü
- **Örnek Scriptler**: `examples/` ve `scripts/` klasörleri
- **Araştırma Makalesi**: (Yakında eklenecek)

## 🆘 Yardım

Sorun yaşıyorsanız:

1. GitHub Issues'a bakın
2. Yeni issue açın (hata raporu veya özellik isteği)
3. Dokümantasyonu kontrol edin

## 📝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Pull request göndermeden önce lütfen:

1. Kodun çalıştığından emin olun
2. Testler yazın (mümkünse)
3. Dokümantasyon ekleyin
4. Kod stiline uyun

---

**Not**: Bu proje aktif geliştirme aşamasında. Özellikler ve API değişebilir.
