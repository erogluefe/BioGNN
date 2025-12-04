# Usage Guide

Complete guide for using the BioGNN multimodal biometric verification system.

## 📋 Table of Contents

1. [Training](#training)
2. [Testing](#testing)
3. [Inference](#inference)
4. [Cross-Validation](#cross-validation)
5. [Web Demo](#web-demo)

---

## 🚀 Training

Train the model on LUTBio dataset:

```bash
# Basic training
python scripts/train_lutbio.py --config configs/lutbio_config.yaml

# Resume from checkpoint
python scripts/train_lutbio.py \
    --config configs/lutbio_config.yaml \
    --resume experiments/lutbio/checkpoints/last.pth
```

### Outputs

- Checkpoints: `experiments/lutbio/checkpoints/`
  - `best_model.pth` - Best model (lowest val EER)
  - `last.pth` - Latest checkpoint
  - `checkpoint_epoch_N.pth` - Periodic checkpoints
- Visualizations: `experiments/lutbio/visualizations/`
  - `training_dashboard.png` - Complete training dashboard
  - `training_curves.png` - Loss/accuracy curves

---

## 🧪 Testing

Evaluate trained model on test set:

```bash
# Test on test set
python scripts/test_lutbio.py \
    --checkpoint experiments/lutbio/checkpoints/best_model.pth

# Test on validation set
python scripts/test_lutbio.py \
    --checkpoint experiments/lutbio/checkpoints/best_model.pth \
    --split val

# Specify output directory
python scripts/test_lutbio.py \
    --checkpoint best_model.pth \
    --split test \
    --output_dir results/test_evaluation
```

### Outputs

- `roc_curve_{split}.png` - ROC curve
- `det_curve_{split}.png` - DET curve
- `score_distribution_{split}.png` - Genuine/impostor score distribution
- `metrics_{split}.txt` - Text summary of metrics
- `results_{split}.npz` - Raw scores for further analysis

### Metrics

- **EER** (Equal Error Rate): Where FAR = FRR
- **FAR** (False Accept Rate): Impostor accepted as genuine
- **FRR** (False Reject Rate): Genuine rejected as impostor
- **AUC**: Area Under ROC Curve
- **Accuracy, Precision, Recall, F1**

---

## 🔍 Inference

Run inference on individual samples:

### Single Sample

```bash
python scripts/inference.py \
    --checkpoint experiments/lutbio/checkpoints/best_model.pth \
    --face path/to/face.jpg \
    --finger path/to/fingerprint.bmp \
    --voice path/to/audio.wav \
    --visualize
```

### Batch Inference (Subject Directory)

```bash
python scripts/inference.py \
    --checkpoint best_model.pth \
    --subject_dir datasets/lutbio/001 \
    --visualize
```

### Options

- `--threshold 0.5` - Decision threshold (default: 0.5)
- `--visualize` - Generate visualization
- `--output_dir ./results` - Output directory

### Example: Using Only Face

```bash
python scripts/inference.py \
    --checkpoint best_model.pth \
    --face image.jpg
```

### Example: Face + Fingerprint

```bash
python scripts/inference.py \
    --checkpoint best_model.pth \
    --face image.jpg \
    --finger fingerprint.bmp \
    --threshold 0.6
```

---

## 🔄 Cross-Validation

Run Leave-One-Subject-Out (LOSO) cross-validation:

```bash
# Run cross-validation
python scripts/cross_validation.py \
    --config configs/lutbio_config.yaml \
    --num_epochs 50

# Specify output directory
python scripts/cross_validation.py \
    --config configs/lutbio_config.yaml \
    --num_epochs 50 \
    --output_dir experiments/cross_validation
```

### What It Does

- Trains N models (N = number of subjects)
- Each model tested on one held-out subject
- Reports per-fold results and mean ± std
- Perfect for small datasets

### Outputs

- `cv_results.txt` - Summary of all folds
- `cv_detailed_results.npz` - Numpy arrays for plotting

### Example Output

```
CROSS-VALIDATION SUMMARY
======================================================================

Per-Fold Results:
----------------------------------------------------------------------
Fold 1 (001): EER=15.23%, Acc=82.50%, AUC=0.8932
Fold 2 (063): EER=12.45%, Acc=85.30%, AUC=0.9123
...

Mean ± Std:
----------------------------------------------------------------------
EER:      14.56% ± 2.34%
Accuracy: 84.12% ± 3.21%
AUC:      0.9015 ± 0.0234
```

---

## 🌐 Web Demo (Gradio)

Launch interactive web interface:

```bash
# Local demo
python demo/gradio_app.py \
    --checkpoint experiments/lutbio/checkpoints/best_model.pth

# Public link (shareable)
python demo/gradio_app.py \
    --checkpoint best_model.pth \
    --share

# Custom port
python demo/gradio_app.py \
    --checkpoint best_model.pth \
    --port 8080
```

### Features

- ✅ Upload face, fingerprint, or voice
- ✅ Real-time prediction
- ✅ Score visualization
- ✅ Adjustable threshold
- ✅ Webcam/microphone support
- ✅ Mobile-friendly interface

### Access

- Local: `http://localhost:7860`
- Share link: Generated if `--share` is used

### Example Usage in Browser

1. Upload a face image
2. Upload a fingerprint (optional)
3. Record or upload voice (optional)
4. Adjust threshold slider if needed
5. Click "🔍 Verify Identity"
6. View result: GENUINE or IMPOSTOR

---

## 📊 Quick Start Examples

### Complete Workflow

```bash
# 1. Train model
python scripts/train_lutbio.py --config configs/lutbio_config.yaml

# 2. Test model
python scripts/test_lutbio.py \
    --checkpoint experiments/lutbio/checkpoints/best_model.pth

# 3. Run cross-validation
python scripts/cross_validation.py \
    --config configs/lutbio_config.yaml \
    --num_epochs 50

# 4. Launch demo
python demo/gradio_app.py \
    --checkpoint experiments/lutbio/checkpoints/best_model.pth \
    --share
```

### Quick Test

```bash
# Test single sample
python scripts/inference.py \
    --checkpoint best_model.pth \
    --face datasets/lutbio/001/face/001_male_56_face_01.jpg \
    --visualize
```

---

## 🐛 Troubleshooting

### Import Error

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/BioGNN"
```

### Gradio Not Found

```bash
pip install gradio
```

### Out of Memory

- Reduce batch size in config
- Use CPU instead of GPU
- Use gradient accumulation

### Poor Performance

- Increase training epochs
- Use more data augmentation
- Try different GNN types (GAT vs GCN)

---

## 💡 Tips

1. **Use multiple modalities** for better accuracy
2. **Adjust threshold** based on security requirements:
   - Lower threshold → More lenient (higher FAR)
   - Higher threshold → More strict (higher FRR)
3. **Cross-validation** is essential for small datasets
4. **Gradio demo** is great for presentations
5. **Save checkpoints** regularly during training

---

## 📚 Additional Resources

- Full presentation: `PRESENTATION.md`
- Configuration guide: `configs/lutbio_config.yaml`
- API documentation: Code docstrings
- GitHub Issues: For bugs and feature requests

---

## 🎯 Next Steps

After completing basic usage:

1. Experiment with hyperparameters
2. Try different modality combinations
3. Implement quality-aware fusion
4. Add more datasets
5. Deploy to production

Enjoy using BioGNN! 🚀
