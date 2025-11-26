# CPU & Mac Kullanım Kılavuzu

Bu dokümantasyon, BioGNN'i **GPU olmadan** (CPU-only) veya **MacBook** üzerinde nasıl kullanacağınızı açıklar.

## 📋 İçindekiler

- [Hızlı Başlangıç](#hızlı-başlangıç)
- [MacBook Kullanıcıları için Özel Notlar](#macbook-kullanıcıları-için-özel-notlar)
- [Performans Optimizasyonları](#performans-optimizasyonları)
- [Sorun Giderme](#sorun-giderme)

## 🚀 Hızlı Başlangıç

### 1. Device Kontrolü

İlk olarak sisteminizin uyumluluğunu kontrol edin:

```bash
python scripts/check_device.py
```

Bu script size:
- Mevcut PyTorch versiyonunu
- Kullanılabilir device'ları (CUDA/MPS/CPU)
- Önerilen ayarları
- Hızlı test sonuçlarını gösterir

### 2. PyTorch Kurulumu (CPU)

**MacOS veya Linux (CPU only):**

```bash
# PyTorch CPU versiyonu
pip install torch torchvision torchaudio

# PyTorch Geometric CPU versiyonu
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Önemli**: CPU için özel wheel dosyalarını kullanın (`+cpu.html` linki).

### 3. Eğitim (CPU)

```bash
# CPU-optimized configuration kullanın
python train.py --config configs/cpu_config.yaml

# Device manuel belirtme (opsiyonel)
python train.py --config configs/cpu_config.yaml --device cpu
```

## 🍎 MacBook Kullanıcıları için Özel Notlar

### Intel Mac (i5, i7, i9)

**Donanım:**
- CPU: Intel Core i5/i7/i9
- Önerilen RAM: 16GB+
- Depolama: SSD önerilir

**Kurulum:**

```bash
# Homebrew ile Python (opsiyonel)
brew install python@3.10

# Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıklar
pip install -r requirements.txt
pip install -e .

# PyTorch Geometric (CPU)
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Eğitim:**

```bash
python train.py --config configs/cpu_config.yaml
```

**Beklenen Performans:**
- Eğitim hızı: ~50-100 samples/sec (batch_size=4)
- Epoch süresi: ~5-15 dakika (veri setine göre)
- GPU'dan 2-5x daha yavaş
- RAM kullanımı: 4-8GB

### Apple Silicon Mac (M1/M2/M3)

**Donanım:**
- CPU: Apple M1/M2/M3
- GPU: Apple Silicon GPU (MPS)
- RAM: 16GB+ (unified memory)

**MPS (Metal Performance Shaders) Desteği:**

PyTorch'un MPS desteği **deneysel** aşamadadır. Bazı operasyonlar CPU'ya geri düşebilir.

**Kurulum:**

```bash
# M1/M2/M3 için ARM64 Python
python3 -m venv venv
source venv/bin/activate

# PyTorch (MPS destekli)
pip3 install torch torchvision torchaudio

# PyTorch Geometric
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Eğitim (MPS ile):**

```bash
# MPS kullanımı
python train.py --config configs/cpu_config.yaml --device mps

# Otomatik algılama (CPU veya MPS)
python train.py --config configs/cpu_config.yaml --device auto
```

**MPS vs CPU:**
- MPS genellikle CPU'dan 1.5-3x daha hızlıdır
- Bazı operasyonlar otomatik olarak CPU'ya düşer
- AMP (Mixed Precision) MPS'de desteklenmez

**Not**: Eğer MPS ile hata alırsanız, CPU moduna geçin:
```bash
python train.py --config configs/cpu_config.yaml --device cpu
```

## ⚡ Performans Optimizasyonları

### 1. Config Dosyası Ayarları

`configs/cpu_config.yaml` dosyası CPU için optimize edilmiştir:

```yaml
# Batch size
batch_size: 4  # Küçük batch size

# Gradient accumulation (batch=32 simülasyonu)
gradient_accumulation_steps: 8

# Hafif model
feature_extractors:
  face:
    backbone: 'resnet18'  # ResNet50 yerine ResNet18

# Küçük görüntü boyutları
data:
  face:
    img_size: [96, 96]  # 112x112 yerine 96x96
```

### 2. PyTorch Threading

CPU kullanımını optimize etmek için thread sayısını ayarlayın:

```python
import torch

# Intel i9 için (8 çekirdek)
torch.set_num_threads(4)  # Çekirdek sayısının yarısı
torch.set_num_interop_threads(2)

# M1/M2/M3 için (8-10 çekirdek)
torch.set_num_threads(6)
torch.set_num_interop_threads(2)
```

Veya environment variable ile:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python train.py --config configs/cpu_config.yaml
```

### 3. MKL-DNN Optimizasyonu (Intel CPU için)

Intel CPU'larda MKL-DNN optimizasyonlarını etkinleştirin:

```python
import torch

# MKL-DNN etkinleştir
torch.backends.mkldnn.enabled = True
```

### 4. DataLoader Optimizasyonları

```yaml
# configs/cpu_config.yaml
data:
  num_workers: 2  # CPU için 2-4 yeterli
  pin_memory: false  # CPU için gereksiz
  prefetch_factor: 2
  persistent_workers: false  # Memory tasarrufu
```

### 5. Model Boyutu Azaltma

```yaml
# Daha küçük hidden dimensions
model:
  feature_dim: 256  # 512 yerine 256
  gnn_hidden_dims: [128, 64]  # [256, 128] yerine

# Daha küçük MFCC features
voice:
  n_mfcc: 20  # 40 yerine
```

## 🐛 Sorun Giderme

### Problem: Out of Memory (OOM)

**Çözüm:**

1. Batch size'ı azaltın:
```yaml
batch_size: 2  # veya 1
gradient_accumulation_steps: 16
```

2. Görüntü boyutlarını azaltın:
```yaml
face:
  img_size: [64, 64]  # 96x96 yerine
```

3. Hafif model kullanın:
```yaml
feature_extractors:
  face:
    backbone: 'resnet18'
```

### Problem: MPS hatası (Apple Silicon)

**Hata**: `RuntimeError: MPS backend out of memory`

**Çözüm:**

1. CPU moduna geçin:
```bash
python train.py --config configs/cpu_config.yaml --device cpu
```

2. Batch size azaltın:
```yaml
batch_size: 2
```

### Problem: Çok yavaş eğitim

**Çözüm:**

1. Daha küçük dataset kullanın (test için):
```python
from biognn.data import SyntheticMultimodalDataset

# Küçük synthetic dataset
dataset = SyntheticMultimodalDataset(
    num_subjects=50,  # 500 yerine 50
    samples_per_subject=3  # 10 yerine 3
)
```

2. Epoch sayısını azaltın:
```yaml
num_epochs: 10  # 50 yerine 10
```

3. Gradient accumulation azaltın:
```yaml
gradient_accumulation_steps: 4  # 8 yerine 4
```

### Problem: PyTorch Geometric import hatası

**Hata**: `ImportError: cannot import name 'GCNConv'`

**Çözüm:**

CPU için PyG'yi tekrar kurun:

```bash
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib torch-geometric

pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## 📊 Beklenen Performans

### Intel MacBook i9 (16GB RAM)

| Metric | Değer |
|--------|-------|
| Batch size | 4 |
| Samples/sec | 50-100 |
| Epoch süresi | 5-15 dk |
| RAM kullanımı | 4-8 GB |
| CPU kullanımı | 60-90% |

### Apple M1/M2 Mac (MPS)

| Metric | Değer |
|--------|-------|
| Batch size | 8-16 |
| Samples/sec | 100-200 |
| Epoch süresi | 3-8 dk |
| RAM kullanımı | 6-10 GB |
| GPU kullanımı | 40-70% |

### Apple M1/M2 Mac (CPU-only)

| Metric | Değer |
|--------|-------|
| Batch size | 4 |
| Samples/sec | 80-150 |
| Epoch süresi | 4-10 dk |
| RAM kullanımı | 4-8 GB |
| CPU kullanımı | 70-95% |

## 💡 İpuçları

1. **İlk test için synthetic data kullanın:**
```bash
python examples/quickstart.py
```

2. **Device compatibility kontrol edin:**
```bash
python scripts/check_device.py
```

3. **Küçük veri setiyle başlayın:**
```bash
python scripts/download_datasets.py --dataset lfw  # ~200MB, hızlı
```

4. **Monitoring:**
```bash
# CPU kullanımı (Mac)
top -o cpu

# Memory kullanımı
top -o mem

# Activity Monitor (GUI)
open -a "Activity Monitor"
```

5. **Gradient accumulation kullanın:**
   - Küçük batch size (4) + accumulation (8) = effective batch 32
   - Memory tasarrufu sağlar
   - GPU kadar etkili değildir ama yardımcı olur

## 🔗 İlgili Dokümantasyon

- [Ana README](../README.md)
- [GETTING_STARTED.md](GETTING_STARTED.md)
- [DATASETS.md](DATASETS.md)

## ❓ Sık Sorulan Sorular

**S: GPU olmadan kullanabilir miyim?**
C: Evet! Proje tamamen CPU modunda çalışır. Sadece daha yavaştır.

**S: MacBook Intel i9 ile ne kadar sürer?**
C: GPU'dan 2-5x daha yavaş. Epoch başına ~5-15 dakika.

**S: Apple Silicon (M1/M2) MPS desteği stabil mi?**
C: Deneysel aşamada. Çoğu işlem çalışır ama bazı hatalar olabilir. CPU fallback önerilir.

**S: Minimum RAM gereksinimi nedir?**
C: 8GB ile çalışır ama 16GB önerilir. Batch size=4 ile 8GB yeterlidir.

**S: Eğitim çok yavaş, ne yapabilirim?**
C:
- Daha küçük model kullanın (ResNet18)
- Batch size azaltın
- Görüntü boyutlarını küçültün
- Daha az epoch kullanın
- Synthetic data ile test edin
