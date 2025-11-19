"""
Training utilities for multimodal biometric models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time

from ..evaluation.metrics import BiometricEvaluator


class Trainer:
    """
    Trainer for multimodal biometric fusion models
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './experiments',
        experiment_name: str = 'default',
        use_amp: bool = True,  # Automatic Mixed Precision
        log_interval: int = 10
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (if None, uses Adam)
            criterion: Loss function (if None, uses CrossEntropyLoss)
            device: Device to train on
            output_dir: Output directory for checkpoints and logs
            experiment_name: Name of experiment
            use_amp: Use automatic mixed precision
            log_interval: Logging interval
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_interval = log_interval
        self.use_amp = use_amp and device == 'cuda'

        # Setup output directory
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        else:
            self.optimizer = optimizer

        # Loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # AMP scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Evaluator
        self.evaluator = BiometricEvaluator()

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_eer': [],
            'learning_rate': []
        }

        # Best model tracking
        self.best_val_eer = float('inf')
        self.best_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')

        for batch_idx, (sample1, sample2, labels) in enumerate(pbar):
            # Move to device
            labels = labels.to(self.device)

            # Prepare modality inputs
            modality_inputs = self._prepare_modality_inputs(sample1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(modality_inputs)
                    loss = self.criterion(logits, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(modality_inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.2f}%'
                })

        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate model

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_predictions = []
        all_scores = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')

            for sample1, sample2, labels in pbar:
                labels = labels.to(self.device)

                # Prepare inputs
                modality_inputs = self._prepare_modality_inputs(sample1)

                # Forward pass
                logits, _ = self.model(modality_inputs)
                loss = self.criterion(logits, labels)

                # Get predictions and scores
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # Collect results
                total_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())  # Score for genuine class

        # Compute metrics
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)

        # Use evaluator for comprehensive metrics
        eval_results = self.evaluator.evaluate(all_labels, all_scores, all_predictions)

        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': eval_results['accuracy'] * 100,
            'eer': eval_results['eer'],
            'auc': eval_results['auc'],
            'far': eval_results['far'],
            'frr': eval_results['frr']
        }

        return metrics

    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Train model for multiple epochs

        Args:
            num_epochs: Number of epochs
            save_best: Save best model based on validation EER
            early_stopping_patience: Early stopping patience (None to disable)
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Use AMP: {self.use_amp}\n")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")

            # Train
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                      f"Val EER: {val_metrics['eer']:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f}")

                # Update learning rate
                self.scheduler.step(val_metrics['loss'])

                # Save history
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_eer'].append(val_metrics['eer'])

                # Save best model
                if save_best and val_metrics['eer'] < self.best_val_eer:
                    self.best_val_eer = val_metrics['eer']
                    self.best_epoch = epoch
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)
                    print(f"✓ Saved best model (EER: {self.best_val_eer:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best EER: {self.best_val_eer:.4f} at epoch {self.best_epoch}")
                    break

            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch)

        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {elapsed_time/60:.2f} minutes")
        if self.val_loader is not None:
            print(f"Best validation EER: {self.best_val_eer:.4f} at epoch {self.best_epoch}")
        print(f"{'='*70}\n")

        # Save final checkpoint and history
        self.save_checkpoint('final_model.pth', num_epochs)
        self.save_history()

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Optional[Dict] = None
    ):
        """
        Save model checkpoint

        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            metrics: Optional metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_eer': self.best_val_eer,
            'history': self.history
        }

        if metrics:
            checkpoint['metrics'] = metrics

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint

        Args:
            filename: Checkpoint filename
        """
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_eer = checkpoint['best_val_eer']
        self.history = checkpoint['history']

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint: {filename}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Best validation EER: {self.best_val_eer:.4f}")

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Saved training history to {history_path}")

    def _prepare_modality_inputs(self, sample) -> Dict[str, torch.Tensor]:
        """
        Prepare modality inputs from a BiometricSample

        Args:
            sample: BiometricSample object

        Returns:
            Dictionary of modality inputs on device
        """
        modality_inputs = {}
        for modality, tensor in sample.modalities.items():
            modality_inputs[modality] = tensor.to(self.device)
        return modality_inputs
