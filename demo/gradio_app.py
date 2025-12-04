#!/usr/bin/env python3
"""
Gradio Web Demo for Multimodal Biometric Verification

Usage:
    python demo/gradio_app.py --checkpoint experiments/lutbio/checkpoints/best_model.pth
    python demo/gradio_app.py --checkpoint best_model.pth --share  # Create public link
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
import gradio as gr

from biognn.fusion import MultimodalBiometricFusion
from biognn.data.lutbio_transforms import get_lutbio_transforms


class BiometricVerificationDemo:
    """Wrapper class for Gradio demo"""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.model, self.config = self.load_checkpoint(checkpoint_path)
        self.transforms = self.get_transforms()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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
        model = model.to(self.device)
        model.eval()

        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")

        return model, config

    def get_transforms(self):
        """Get preprocessing transforms"""
        return get_lutbio_transforms(
            split='val',
            image_size=self.config['dataset']['face_size'],
            fingerprint_size=self.config['dataset']['fingerprint_size'],
            spectrogram_size=tuple(self.config['dataset']['spectrogram_size']),
            augmentation=False
        )

    @torch.no_grad()
    def predict(
        self,
        face_img,
        finger_img,
        voice_audio,
        threshold: float = 0.5
    ):
        """
        Make prediction

        Args:
            face_img: PIL Image or None
            finger_img: PIL Image or None
            voice_audio: Audio file path or None
            threshold: Decision threshold

        Returns:
            Tuple of (prediction_text, confidence_text, score)
        """
        try:
            modalities = {}

            # Process face
            if face_img is not None:
                face_img = Image.fromarray(face_img).convert('RGB')
                face_tensor = self.transforms['face'](face_img)
                modalities['face'] = face_tensor.unsqueeze(0).to(self.device)

            # Process finger
            if finger_img is not None:
                finger_img = Image.fromarray(finger_img).convert('RGB')
                finger_tensor = self.transforms['finger'](finger_img)
                modalities['finger'] = finger_tensor.unsqueeze(0).to(self.device)

            # Process voice
            if voice_audio is not None:
                waveform, sample_rate = torchaudio.load(voice_audio)
                voice_tensor = self.transforms['voice'](waveform)
                modalities['voice'] = voice_tensor.unsqueeze(0).to(self.device)

            # Check if at least one modality is provided
            if not modalities:
                return (
                    "❌ Error: Please provide at least one modality",
                    "",
                    None
                )

            # Forward pass
            logits, _ = self.model(modalities)
            score = torch.sigmoid(logits.squeeze()).item()

            # Make decision
            is_genuine = score >= threshold
            prediction = "✅ GENUINE" if is_genuine else "❌ IMPOSTOR"

            # Calculate confidence
            confidence = abs(score - 0.5) * 2  # 0-1 scale
            confidence_pct = confidence * 100

            # Create detailed result
            result_text = f"""
## {prediction}

**Score:** {score:.4f}
**Threshold:** {threshold:.4f}
**Confidence:** {confidence_pct:.1f}%

### Modalities Used:
{', '.join(modalities.keys()).upper()}
            """

            # Additional info
            info_text = f"""
**Model Configuration:**
- GNN Type: {self.config['model']['gnn_type'].upper()}
- Feature Dim: {self.config['model']['feature_dim']}
- Modalities: {', '.join(self.config['dataset']['modalities'])}
            """

            return result_text, info_text, score

        except Exception as e:
            return f"❌ Error: {str(e)}", "", None


def create_demo(checkpoint_path: str, device: torch.device):
    """Create Gradio interface"""

    # Initialize demo
    demo_model = BiometricVerificationDemo(checkpoint_path, device)

    # Custom CSS
    custom_css = """
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """

    # Create interface
    with gr.Blocks(css=custom_css, title="Multimodal Biometric Verification") as demo:

        # Header
        gr.Markdown("""
        <div class="header">
            <h1>🔐 Multimodal Biometric Verification</h1>
            <p>Graph Neural Network-based Multimodal Fusion System</p>
            <p>Upload face, fingerprint, or voice samples for verification</p>
        </div>
        """)

        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Modalities")
                gr.Markdown("*Upload at least one modality*")

                face_input = gr.Image(
                    label="Face Image",
                    type="numpy",
                    sources=["upload", "webcam"]
                )

                finger_input = gr.Image(
                    label="Fingerprint Image",
                    type="numpy",
                    sources=["upload"]
                )

                voice_input = gr.Audio(
                    label="Voice Audio",
                    type="filepath",
                    sources=["upload", "microphone"]
                )

                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="Decision Threshold",
                    info="Score threshold for genuine/impostor decision"
                )

                verify_btn = gr.Button(
                    "🔍 Verify Identity",
                    variant="primary",
                    size="lg"
                )

                clear_btn = gr.Button("🔄 Clear")

            # Right column: Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Verification Result")

                result_output = gr.Markdown(
                    label="Result",
                    value="*Awaiting input...*"
                )

                info_output = gr.Markdown(
                    label="Model Info",
                    value=""
                )

                score_plot = gr.Plot(label="Score Visualization")

        # Examples section
        gr.Markdown("### 📋 Example Usage")
        gr.Markdown("""
        1. **Single Modality**: Upload just a face image
        2. **Dual Modality**: Upload face + fingerprint
        3. **Triple Modality**: Upload all three modalities for best accuracy
        4. **Adjust Threshold**: Lower threshold = more lenient (higher FAR), Higher threshold = more strict (higher FRR)
        """)

        # Info section
        with gr.Accordion("ℹ️ About This System", open=False):
            gr.Markdown("""
            ## Multimodal Biometric Verification System

            This system uses **Graph Neural Networks (GNN)** to fuse multiple biometric modalities:

            ### 🧠 Model Architecture
            - **Feature Extraction**: CNN-based extractors for each modality
            - **Graph Construction**: Modalities as nodes, relationships as edges
            - **GNN Fusion**: Graph Attention Network (GAT) for adaptive fusion
            - **Decision**: Binary verification (Genuine vs Impostor)

            ### 📊 Metrics
            - **EER** (Equal Error Rate): Lower is better
            - **FAR** (False Accept Rate): Impostor accepted as genuine
            - **FRR** (False Reject Rate): Genuine rejected as impostor

            ### 🎯 Performance
            - Trained on LUTBio multimodal biometric database
            - Supports face, fingerprint, and voice modalities
            - Adaptive fusion based on modality quality

            ### 🔒 Privacy Note
            - All processing happens locally
            - No data is stored or transmitted
            - Model runs on-device
            """)

        # Create score visualization function
        def visualize_score(score):
            """Create score visualization"""
            if score is None:
                return None

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 2))

            # Draw score bar
            colors = ['red', 'yellow', 'green']
            positions = [0, 0.5, 1.0]

            # Background gradient
            for i in range(len(colors) - 1):
                ax.axhspan(0, 1, positions[i], positions[i+1],
                          facecolor=colors[i], alpha=0.3)

            # Score marker
            ax.plot([score, score], [0, 1], 'b-', linewidth=5,
                   label=f'Score: {score:.3f}')

            # Threshold line
            ax.axvline(0.5, color='black', linestyle='--', linewidth=2,
                      label='Threshold: 0.5')

            # Labels
            ax.text(0.25, 0.5, 'IMPOSTOR', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='red')
            ax.text(0.75, 0.5, 'GENUINE', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='green')

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('Verification Score', fontsize=12)
            ax.set_yticks([])
            ax.legend(loc='upper right')
            ax.set_title('Verification Score', fontsize=14, fontweight='bold')

            plt.tight_layout()
            return fig

        # Define verification function
        def verify(face, finger, voice, threshold):
            """Verification wrapper"""
            result, info, score = demo_model.predict(face, finger, voice, threshold)
            plot = visualize_score(score)
            return result, info, plot

        # Connect button
        verify_btn.click(
            fn=verify,
            inputs=[face_input, finger_input, voice_input, threshold_slider],
            outputs=[result_output, info_output, score_plot]
        )

        # Clear button
        clear_btn.click(
            fn=lambda: (None, None, None, 0.5, "*Awaiting input...*", "", None),
            outputs=[face_input, finger_input, voice_input, threshold_slider,
                    result_output, info_output, score_plot]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description='Gradio demo for biometric verification')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--share', action='store_true',
                       help='Create public link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port number (default: 7860)')
    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    # Create demo
    print("\nCreating Gradio interface...")
    demo = create_demo(args.checkpoint, device)

    # Launch
    print(f"\n{'='*60}")
    print("LAUNCHING DEMO")
    print(f"{'='*60}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print(f"{'='*60}\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == '__main__':
    main()
