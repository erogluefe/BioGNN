# Veri Seti Kılavuzu

Bu dokümantasyon, BioGNN ile kullanılabilecek multimodal biyometrik veri setlerini ve bunları nasıl hazırlayacağınızı açıklar.

## 📥 Otomatik Veri Seti İndirme

BioGNN, birçok popüler biyometrik veri setini otomatik olarak indirmek için yerleşik downloader'lar sağlar.

### Hızlı Başlangıç

```bash
# Tüm mevcut veri setlerini listele
python scripts/download_datasets.py --list

# Belirli bir veri setini indir
python scripts/download_datasets.py --dataset lfw --root ./datasets

# Birden fazla veri setini indir
python scripts/download_datasets.py --dataset lfw socofing librispeech --root ./datasets

# LibriSpeech için özel subset
python scripts/download_datasets.py --dataset librispeech --subset dev-clean
```

### Python'dan Kullanım

```python
from biognn.data.downloaders import get_downloader

# LFW veri setini indir
downloader = get_downloader('lfw', root='./datasets')
dataset_path = downloader.download()

# LibriSpeech subset indir
from biognn.data.downloaders import LibriSpeechDownloader
downloader = LibriSpeechDownloader(root='./datasets', subset='dev-clean')
dataset_path = downloader.download()

# SOCOFing (Kaggle - API credentials gerekli)
from biognn.data.downloaders import SOCOFingDownloader
downloader = SOCOFingDownloader(root='./datasets')
dataset_path = downloader.download()
```

### Otomatik İndirme Destekleyen Veri Setleri

| Veri Seti | Boyut | İndirme Türü | Ek Gereksinim |
|-----------|-------|--------------|---------------|
| **LFW** | ~200MB | Otomatik | Yok |
| **CelebA** | ~1.3GB | Otomatik | Google Drive (manuel gerekebilir) |
| **SOCOFing** | ~1GB | Kaggle API | Kaggle credentials |
| **LibriSpeech** | 340MB-60GB | Otomatik | Yok |

### Manuel İndirme Gerektiren Veri Setleri

Bazı veri setleri kayıt ve anlaşma gerektirdiği için manuel indirme talimatları gösterilir:

```bash
# Talimatleri göster
python scripts/download_datasets.py --dataset casia-webface --show-instructions
python scripts/download_datasets.py --dataset voxceleb --show-instructions
python scripts/download_datasets.py --dataset fvc2004 --show-instructions
```

### Kaggle Veri Setleri İçin Kurulum

SOCOFing gibi Kaggle veri setleri için:

```bash
# Kaggle API'yi kur
pip install kaggle

# Kaggle credentials yapılandır
# 1. https://www.kaggle.com/settings/account adresine git
# 2. "Create New API Token" tıkla
# 3. kaggle.json dosyasını ~/.kaggle/ dizinine yerleştir
# 4. İzinleri ayarla
chmod 600 ~/.kaggle/kaggle.json

# Artık Kaggle veri setlerini indirebilirsiniz
python scripts/download_datasets.py --dataset socofing
```

## 🗂️ Önerilen Açık Veri Setleri

### 1. Yüz Tanıma

#### CASIA-WebFace
- **Açıklama**: 10,575 kişiye ait 494,414 yüz görüntüsü
- **İndirme**: http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
- **Lisans**: Research use only
- **Format**: JPEG images
- **Kullanım**: Yüz doğrulama ve tanıma için en popüler veri setlerinden biri

#### LFW (Labeled Faces in the Wild)
- **Açıklama**: 5,749 kişiye ait 13,233 görüntü
- **İndirme**: http://vis-www.cs.umass.edu/lfw/
- **Lisans**: Public domain
- **Kullanım**: Benchmark için ideal

#### CelebA
- **Açıklama**: 10,177 ünlüye ait 202,599 görüntü
- **İndirme**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Lisans**: Non-commercial research purposes
- **Özellikler**: 40 binary attribute annotation

### 2. Parmak İzi

#### FVC2004 (Fingerprint Verification Competition)
- **Açıklama**: 4 farklı veri seti (DB1-DB4)
- **İndirme**: http://bias.csr.unibo.it/fvc2004/
- **Lisans**: Research use
- **Format**: TIFF images (500 DPI)
- **Kullanım**: Parmak izi doğrulama benchmark'ı

#### SOCOFing
- **Açıklama**: 6,000 parmak izi görüntüsü
- **İndirme**: https://www.kaggle.com/datasets/ruizgara/socofing
- **Lisans**: Kaggle license
- **Format**: BMP images
- **Özellikler**: Gerçek ve değiştirilmiş parmak izleri

### 3. Iris Tanıma

#### CASIA-Iris-V4
- **Açıklama**: Birden fazla iris veri seti (Interval, Lamp, Twins, Distance, Synthetic)
- **İndirme**: http://biometrics.idealtest.org/
- **Lisans**: Research use only
- **Format**: JPEG images
- **Özellikler**: Farklı yakalama koşulları

#### UBIRIS
- **Açıklama**: 241 kişiye ait iris görüntüleri
- **İndirme**: http://iris.di.ubi.pt/
- **Lisans**: Free for research
- **Format**: JPEG images
- **Özellikler**: Gürültülü ve temiz versiyonlar

### 4. Ses/Konuşmacı Tanıma

#### VoxCeleb1/2
- **Açıklama**: 7,000+ konuşmacıya ait 1M+ ses kayıtları
- **İndirme**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- **Lisans**: Free for research
- **Format**: M4A audio files
- **Kullanım**: Konuşmacı doğrulama ve tanıma

#### LibriSpeech
- **Açıklama**: 1,000 saatlik İngilizce konuşma
- **İndirme**: http://www.openslr.org/12
- **Lisans**: CC BY 4.0
- **Format**: FLAC audio files

## 📁 Veri Organizasyonu

### Önerilen Dizin Yapısı

```
datasets/
├── train/
│   ├── subject_001/
│   │   ├── face_001.jpg        # İlk yüz örneği
│   │   ├── face_002.jpg        # İkinci yüz örneği
│   │   ├── fingerprint_001.png # Parmak izi
│   │   ├── iris_001.png        # Iris
│   │   └── voice_001.wav       # Ses kaydı
│   ├── subject_002/
│   │   └── ...
│   └── ...
├── val/
│   └── ... (aynı yapı)
└── test/
    └── ... (aynı yapı)
```

### Dosya Adlandırma Kuralları

- **Format**: `{modality}_{index}.{extension}`
- **Modalite isimleri**: `face`, `fingerprint`, `iris`, `voice`
- **Index**: 3 haneli numara (001, 002, ...)
- **Extension**: jpg/png (görüntü), wav/flac (ses)

### Örnek

```
subject_042/
├── face_001.jpg
├── face_002.jpg
├── face_003.jpg
├── fingerprint_001.png
├── fingerprint_002.png
├── iris_001.png
└── voice_001.wav
```

## 🔧 Veri Preprocessing

### Yüz Görüntüleri

```python
from biognn.data import FaceTransform

transform = FaceTransform(
    img_size=(112, 112),
    augment=True,  # Training için
    normalize=True
)

# Kullanım
face_img = Image.open('face.jpg')
face_tensor = transform(face_img)  # [3, 112, 112]
```

### Parmak İzi

```python
from biognn.data import FingerprintTransform

transform = FingerprintTransform(
    img_size=(96, 96),
    augment=True,
    normalize=True
)
```

### Iris

```python
from biognn.data import IrisTransform

transform = IrisTransform(
    img_size=(64, 256),  # Unwrapped iris boyutu
    augment=True,
    normalize=True
)
```

### Ses

```python
from biognn.data import VoiceTransform

transform = VoiceTransform(
    sample_rate=16000,
    n_mfcc=40,
    n_fft=512,
    hop_length=160,
    augment=True,
    max_length=16000*3  # 3 saniye
)

# MFCC features çıkarır: [40, time_frames]
```

## 📝 Veri Seti Oluşturma

### 1. Kendi Verilerinizi Organize Edin

```bash
# Script kullanarak organize edin
python scripts/organize_dataset.py \
    --input /path/to/raw/data \
    --output ./datasets \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### 2. Dataset Sınıfı Oluşturun

```python
from biognn.data import MultimodalBiometricDataset

class MyDataset(MultimodalBiometricDataset):
    def _load_data(self):
        # Veri yükleme mantığı
        pass
    
    def __getitem__(self, idx):
        # Bir örnek döndür
        pass
```

### 3. Doğrulama

```python
# Veri setini test edin
dataset = MyDataset(root='./datasets', split='train')
print(f"Dataset size: {len(dataset)}")

# İlk örneği kontrol edin
sample = dataset[0]
print(f"Subject ID: {sample.subject_id}")
print(f"Modalities: {sample.get_available_modalities()}")

for mod, data in sample.modalities.items():
    print(f"  {mod}: {data.shape}")
```

## 🎯 Veri Artırma (Augmentation)

### Eğitim İçin Öneriler

```python
# Agresif augmentation
train_transforms = {
    'face': FaceTransform(augment=True),
    'fingerprint': FingerprintTransform(augment=True),
    'iris': IrisTransform(augment=True),
    'voice': VoiceTransform(augment=True)
}

# Validation/test için augmentation YOK
val_transforms = {
    'face': FaceTransform(augment=False),
    'fingerprint': FingerprintTransform(augment=False),
    'iris': IrisTransform(augment=False),
    'voice': VoiceTransform(augment=False)
}
```

## 📊 Veri İstatistikleri

Veri setinizi analiz edin:

```python
# Dataset istatistikleri
from collections import Counter

subject_counts = Counter()
modality_counts = {mod: 0 for mod in ['face', 'fingerprint', 'iris', 'voice']}

for idx in range(len(dataset)):
    sample = dataset[idx]
    subject_counts[sample.subject_id] += 1
    
    for mod in sample.get_available_modalities():
        modality_counts[mod] += 1

print(f"Unique subjects: {len(subject_counts)}")
print(f"Avg samples per subject: {np.mean(list(subject_counts.values())):.2f}")
print(f"Modality coverage:")
for mod, count in modality_counts.items():
    print(f"  {mod}: {count} ({count/len(dataset)*100:.1f}%)")
```

## ⚠️ Yaygın Hatalar ve Çözümleri

### 1. Dosya Bulunamadı

**Hata**: `FileNotFoundError: Data directory not found`

**Çözüm**: Dizin yapısını kontrol edin:
```bash
ls -R datasets/train/subject_001/
```

### 2. Boyut Uyumsuzluğu

**Hata**: `RuntimeError: Expected 3D tensor, got 2D`

**Çözüm**: Transform'ları doğru uygulayın:
```python
# Yanlış
img = np.array(Image.open('face.jpg'))  # NumPy array

# Doğru
img = Image.open('face.jpg')  # PIL Image
img = transform(img)  # Tensor'a dönüştürülür
```

### 3. Ses Format Hatası

**Hata**: `soundfile.LibsndfileError: Format not recognised`

**Çözüm**: Ses dosyasını dönüştürün:
```bash
# ffmpeg kullanarak WAV'a dönüştür
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 📚 Ek Kaynaklar

- **Veri artırma teknikleri**: `biognn/data/transforms.py`
- **Feature extraction**: `biognn/data/feature_extractors.py`
- **Örnek dataset**: `biognn/data/example_dataset.py`

## 🔗 Linkler

- NIST Biometric Datasets: https://www.nist.gov/itl/iad/image-group/biometric-data
- IEEE Biometrics Council: https://ieee-biometrics.org/
- Biometric Evaluation: https://www.iso.org/standard/78160.html
