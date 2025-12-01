# LUTBio Dataset Integration Guide

Complete guide for using BioGNN with LUTBio multimodal biometric database.

## 📊 Dataset Overview

**LUTBio** (Lappeenranta University of Technology Biometric Database)
- **306 subjects** (164 male, 142 female)
- **Age range**: 8-90 years
- **9 modalities**: Face, Fingerprint, Voice, Palmprint, ECG, Ear, Periocular, Opisthenar
- **License**: CC BY 4.0
- **DOI**: 10.17632/jszw485f8j.6

### BioGNN Support

Currently implemented modalities:
- ✅ **Face** (6 images per subject)
- ✅ **Fingerprint** (10 images per subject)
- ✅ **Voice** (3 audio files per subject)

## 🚀 Quick Start

### 1. Dataset Setup

```bash
# Download LUTBio dataset
# Visit: https://data.mendeley.com/datasets/jszw485f8j/6
# Fill the application form and email to: rykeryang@163.com

# Once downloaded, extract to datasets directory
mkdir -p datasets/lutbio
unzip "LUTBIO sample data.zip" -d datasets/lutbio/
```

Expected structure:
```
datasets/lutbio/
├── 001/
│   ├── face/
│   ├── finger/
│   └── voice/
├── 063/
├── 120/
...
```

### 2. Visualize Dataset

```bash
# Visualize real LUTBio samples
python examples/visualize_lutbio.py --root datasets/lutbio

# Output: outputs/lutbio/
#   - lutbio_sample_grid.png
#   - lutbio_distribution.png
#   - lutbio_genuine_vs_imposter.png
```

### 3. Training

```bash
# Train with default config (optimized for 6 subjects demo)
python scripts/train_lutbio.py --config configs/lutbio_config.yaml

# Monitor training
# Outputs saved to: experiments/lutbio/
```

### 4. View Results

```bash
# Training curves and metrics
open experiments/lutbio/visualizations/training_dashboard.png

# Model checkpoints
ls experiments/lutbio/checkpoints/
#   - best_model.pth (lowest EER)
#   - last.pth (latest epoch)
```

## 📝 Configuration

Edit `configs/lutbio_config.yaml` for custom settings:

```yaml
# Dataset
dataset:
  root: "datasets/lutbio"
  modalities: ['face', 'finger', 'voice']
  pairs_per_subject: 20  # Increase for more data

# Model
model:
  gnn_type: "gat"  # or 'gcn', 'graphsage'
  feature_dim: 512

# Training
training:
  num_epochs: 100
  batch_size: 8  # Adjust based on your hardware
```

## 🎯 Demo Dataset (6 Subjects)

Default split for demo:
- **Train**: 4 subjects (001, 063, 120, 162)
- **Val**: 1 subject (273)
- **Test**: 1 subject (303)

### Custom Split

```python
from biognn.data.lutbio_dataset import LUTBioDataset

dataset = LUTBioDataset(
    root='datasets/lutbio',
    modalities=['face', 'finger', 'voice'],
    split='train',
    train_subjects=['001', '063'],
    val_subjects=['120'],
    test_subjects=['162']
)
```

## 📊 Evaluation Metrics

The system computes:

### Verification (1:1)
- **EER** (Equal Error Rate) - Primary metric
- **FAR** (False Accept Rate)
- **FRR** (False Reject Rate)
- **AUC** (Area Under ROC Curve)

### Identification (1:N)
- **Rank-1 Accuracy**
- **Rank-5 Accuracy**
- **CMC Curve** (Cumulative Match Characteristic)

## 🔧 Advanced Usage

### Custom Modality Selection

```python
# Only face and fingerprint
dataset = LUTBioDataset(
    root='datasets/lutbio',
    modalities=['face', 'finger'],  # Exclude voice
    split='train'
)
```

### Custom Transforms

```python
from biognn.data.lutbio_transforms import get_lutbio_transforms

transforms = get_lutbio_transforms(
    split='train',
    face_size=224,  # Larger face images
    fingerprint_size=128,
    augmentation=True
)
```

### Verification vs Identification

```python
# Verification mode (pairs)
ver_dataset = LUTBioDataset(
    root='datasets/lutbio',
    mode='verification',
    pairs_per_subject=30
)

# Identification mode (gallery/probe)
id_dataset = LUTBioDataset(
    root='datasets/lutbio',
    mode='identification'
)
```

## 📈 Expected Performance (Demo Data)

With 6 subjects (very limited):
- **EER**: ~5-15% (limited data)
- **Rank-1 Accuracy**: ~60-80%

With full dataset (306 subjects):
- **EER**: <2% (expected)
- **Rank-1 Accuracy**: >95% (expected)

## 🐛 Troubleshooting

### Dataset Not Found
```bash
# Check path
ls datasets/lutbio/001/face/
# Should show: 001_male_56_face_01.jpg, etc.
```

### Memory Error
```yaml
# Reduce batch size in config
training:
  batch_size: 4  # Instead of 8
```

### Slow Training
```yaml
# Disable augmentation for faster training
dataset:
  augmentation: false
```

## 📚 Dataset Citation

If you use LUTBio dataset, please cite:

```bibtex
@data{jszw485f8j-6,
  title = {LUTBIO Multimodal Biometric Database},
  author = {Yang, Ruikai and others},
  publisher = {Mendeley Data},
  year = {2025},
  doi = {10.17632/jszw485f8j.6}
}
```

## 🔗 Related Files

- Dataset Loader: `biognn/data/lutbio_dataset.py`
- Transforms: `biognn/data/lutbio_transforms.py`
- Config: `configs/lutbio_config.yaml`
- Training: `scripts/train_lutbio.py`
- Visualization: `examples/visualize_lutbio.py`

## ❓ FAQ

**Q: Can I use only 2 modalities?**
A: Yes! Just specify in config: `modalities: ['face', 'finger']`

**Q: How to increase training data?**
A: Increase `pairs_per_subject` in config or get full dataset (306 subjects)

**Q: Can I use my own biometric data?**
A: Yes! Create a custom dataset following the LUTBio structure

**Q: How to improve performance?**
A: 1) Get full dataset, 2) Increase epochs, 3) Tune hyperparameters

---

For more information, see main [README.md](../README.md)
