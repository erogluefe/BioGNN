#!/usr/bin/env python3
"""
Inference script for multimodal biometric verification

Usage:
    # Single verification
    python scripts/inference.py \
        --checkpoint experiments/lutbio/checkpoints/best_model.pth \
        --face path/to/face.jpg \
        --finger path/to/finger.bmp \
        --voice path/to/voice.wav

    # Batch inference from directory
    python scripts/inference.py \
        --checkpoint best_model.pth \
        --subject_dir datasets/lutbio/001 \
        --visualize
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchaudio
import matplotlib.pyplot as plt

from biognn.fusion import MultimodalBiometricFusion
from biognn.data.lutbio_transforms import get_lutbio_transforms


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    # Build model
    gnn_config = config['model'].get('gnn_config', {}).copy()
    gnn_config['num_classes'] = config['model'].get('num_classes', 2)

    model = MultimodalBiometricFusion(
        modalities=config['dataset']['modalities'],
        feature_dim=config['model']['feature_dim'],
        gnn_type=config['model']['gnn_type'],
        gnn_config=gnn_config,
        edge_strategy=config['model']['graph']['edge_strategy'],
        use_adaptive_edges=config['model']['graph']['use_adaptive_edges'],
        use_quality_scores=config['model']['graph']['use_quality_scores']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")

    return model, config


def load_sample(
    face_path: str = None,
    finger_path: str = None,
    voice_path: str = None,
    config: dict = None
) -> dict:
    """
    Load and preprocess a single sample

    Args:
        face_path: Path to face image
        finger_path: Path to fingerprint image
        voice_path: Path to voice audio
        config: Model configuration

    Returns:
        Dictionary of preprocessed modality tensors
    """
    # Get transforms
    transforms = get_lutbio_transforms(
        split='val',
        image_size=config['dataset']['face_size'],
        fingerprint_size=config['dataset']['fingerprint_size'],
        spectrogram_size=tuple(config['dataset']['spectrogram_size']),
        augmentation=False
    )

    modalities = {}

    # Load face
    if face_path:
        print(f"Loading face: {face_path}")
        face_img = Image.open(face_path).convert('RGB')
        face_tensor = transforms['face'](face_img)
        modalities['face'] = face_tensor.unsqueeze(0)  # Add batch dimension

    # Load finger
    if finger_path:
        print(f"Loading fingerprint: {finger_path}")
        finger_img = Image.open(finger_path).convert('RGB')
        finger_tensor = transforms['finger'](finger_img)
        modalities['finger'] = finger_tensor.unsqueeze(0)

    # Load voice
    if voice_path:
        print(f"Loading voice: {voice_path}")
        waveform, sample_rate = torchaudio.load(voice_path)
        voice_tensor = transforms['voice'](waveform)
        modalities['voice'] = voice_tensor.unsqueeze(0)

    if not modalities:
        raise ValueError("At least one modality must be provided")

    return modalities


@torch.no_grad()
def predict(
    model: nn.Module,
    modalities: dict,
    device: torch.device,
    threshold: float = 0.5
) -> dict:
    """
    Make prediction for a sample

    Args:
        model: Trained model
        modalities: Dictionary of modality tensors
        device: Device
        threshold: Decision threshold

    Returns:
        Dictionary with prediction results
    """
    model.eval()

    # Move to device
    for key in modalities:
        modalities[key] = modalities[key].to(device)

    # Forward pass
    logits, attention_weights = model(modalities)

    # Convert to probability
    score = torch.sigmoid(logits.squeeze()).item()

    # Make decision
    prediction = "GENUINE" if score >= threshold else "IMPOSTOR"

    # Extract attention weights if available
    if attention_weights is not None:
        attention_dict = {}
        modality_names = list(modalities.keys())
        for i, mod1 in enumerate(modality_names):
            for j, mod2 in enumerate(modality_names):
                if i != j:
                    key = f"{mod1} → {mod2}"
                    # attention_weights shape depends on GNN implementation
                    # This is a simplified version
                    attention_dict[key] = 0.5  # Placeholder
    else:
        attention_dict = None

    return {
        'prediction': prediction,
        'score': score,
        'logit': logits.squeeze().item(),
        'confidence': abs(score - 0.5) * 2,  # 0-1 scale
        'threshold': threshold,
        'attention': attention_dict
    }


def visualize_prediction(
    modalities: dict,
    result: dict,
    save_path: Path = None
):
    """
    Visualize prediction with modalities

    Args:
        modalities: Input modalities (as tensors)
        result: Prediction result
        save_path: Optional path to save figure
    """
    num_modalities = len(modalities)
    fig, axes = plt.subplots(1, num_modalities + 1, figsize=(5 * (num_modalities + 1), 5))

    if num_modalities == 1:
        axes = [axes]

    # Plot each modality
    idx = 0
    for modality, tensor in modalities.items():
        ax = axes[idx]

        # Remove batch dimension and move to CPU
        img = tensor.squeeze(0).cpu()

        if modality == 'face':
            # Denormalize face image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title('Face', fontsize=12)

        elif modality == 'finger':
            # Denormalize fingerprint
            img = img * 0.5 + 0.5
            img = torch.clamp(img, 0, 1)
            img = img.squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title('Fingerprint', fontsize=12)

        elif modality == 'voice':
            # Plot spectrogram
            spec = img.numpy()
            ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title('Voice Spectrogram', fontsize=12)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')

        ax.axis('off')
        idx += 1

    # Plot prediction result
    ax_result = axes[idx]
    ax_result.axis('off')

    # Create result text
    pred_color = 'green' if result['prediction'] == 'GENUINE' else 'red'
    result_text = f"""
    Prediction: {result['prediction']}

    Score: {result['score']:.4f}
    Threshold: {result['threshold']:.4f}
    Confidence: {result['confidence']*100:.1f}%

    {result['prediction']}
    """

    ax_result.text(0.5, 0.5, result_text,
                  fontsize=14,
                  ha='center',
                  va='center',
                  bbox=dict(boxstyle='round',
                           facecolor=pred_color,
                           alpha=0.3))

    plt.suptitle('Biometric Verification Result', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")

    return fig


def load_subject_samples(subject_dir: Path, config: dict) -> list:
    """
    Load all samples from a subject directory

    Args:
        subject_dir: Path to subject directory
        config: Model configuration

    Returns:
        List of modality dictionaries
    """
    samples = []

    # Find all face images
    face_dir = subject_dir / 'face'
    if face_dir.exists():
        face_files = sorted(face_dir.glob('*.jpg'))
    else:
        face_files = []

    # Find all finger images
    finger_dir = subject_dir / 'finger'
    if finger_dir.exists():
        finger_files = sorted(finger_dir.glob('*.bmp'))
    else:
        finger_files = []

    # Find all voice files
    voice_dir = subject_dir / 'voice'
    if voice_dir.exists():
        voice_files = sorted(voice_dir.glob('*.wav'))
    else:
        voice_files = []

    # Create combinations
    max_samples = max(len(face_files), len(finger_files), len(voice_files))

    for i in range(max_samples):
        sample = load_sample(
            face_path=str(face_files[i % len(face_files)]) if face_files else None,
            finger_path=str(finger_files[i % len(finger_files)]) if finger_files else None,
            voice_path=str(voice_files[i % len(voice_files)]) if voice_files else None,
            config=config
        )
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description='Biometric verification inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--face', type=str, default=None,
                       help='Path to face image')
    parser.add_argument('--finger', type=str, default=None,
                       help='Path to fingerprint image')
    parser.add_argument('--voice', type=str, default=None,
                       help='Path to voice audio file')
    parser.add_argument('--subject_dir', type=str, default=None,
                       help='Path to subject directory (batch mode)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold (default: 0.5)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory for visualizations')
    args = parser.parse_args()

    # Check inputs
    if not (args.face or args.finger or args.voice or args.subject_dir):
        print("Error: At least one modality or subject_dir must be provided")
        return

    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}\n")

    # Load model
    model, config = load_checkpoint(args.checkpoint, device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single sample mode
    if args.face or args.finger or args.voice:
        print("\n" + "="*60)
        print("SINGLE SAMPLE INFERENCE")
        print("="*60 + "\n")

        # Load sample
        modalities = load_sample(
            face_path=args.face,
            finger_path=args.finger,
            voice_path=args.voice,
            config=config
        )

        print(f"\nLoaded modalities: {list(modalities.keys())}")

        # Make prediction
        result = predict(model, modalities, device, args.threshold)

        # Print results
        print("\n" + "-"*60)
        print("PREDICTION RESULT")
        print("-"*60)
        print(f"  Prediction:  {result['prediction']}")
        print(f"  Score:       {result['score']:.4f}")
        print(f"  Threshold:   {result['threshold']:.4f}")
        print(f"  Confidence:  {result['confidence']*100:.1f}%")
        print("-"*60 + "\n")

        # Visualize
        if args.visualize:
            fig = visualize_prediction(
                modalities,
                result,
                save_path=output_dir / 'prediction.png'
            )
            plt.show()

    # Batch mode (subject directory)
    elif args.subject_dir:
        print("\n" + "="*60)
        print("BATCH INFERENCE")
        print("="*60 + "\n")

        subject_dir = Path(args.subject_dir)
        if not subject_dir.exists():
            print(f"Error: Subject directory not found: {subject_dir}")
            return

        # Load all samples
        samples = load_subject_samples(subject_dir, config)
        print(f"Loaded {len(samples)} samples from {subject_dir.name}\n")

        # Run predictions
        results = []
        for i, modalities in enumerate(samples):
            result = predict(model, modalities, device, args.threshold)
            results.append(result)

            print(f"Sample {i+1}: {result['prediction']} (score: {result['score']:.4f})")

            if args.visualize:
                fig = visualize_prediction(
                    modalities,
                    result,
                    save_path=output_dir / f'sample_{i+1}.png'
                )
                plt.close(fig)

        # Summary statistics
        genuine_count = sum(1 for r in results if r['prediction'] == 'GENUINE')
        impostor_count = len(results) - genuine_count
        avg_score = np.mean([r['score'] for r in results])

        print("\n" + "-"*60)
        print("SUMMARY")
        print("-"*60)
        print(f"  Total samples:    {len(results)}")
        print(f"  Genuine:          {genuine_count}")
        print(f"  Impostor:         {impostor_count}")
        print(f"  Average score:    {avg_score:.4f}")
        print("-"*60 + "\n")

    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
