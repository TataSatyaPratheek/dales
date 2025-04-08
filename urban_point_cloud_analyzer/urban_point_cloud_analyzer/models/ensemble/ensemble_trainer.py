# urban_point_cloud_analyzer/models/ensemble/ensemble_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from urban_point_cloud_analyzer.models.ensemble.model_ensemble import SegmentationEnsemble, MultiScaleEnsemble, SpecialistEnsemble

class EnsembleTrainer:
    """
    Trainer for ensemble models that handles the complexities of training multiple models together
    or using pre-trained models in an ensemble.
    """
    
    def __init__(self, 
                 ensemble_model: nn.Module,
                 config: Dict,
                 device: torch.device):
        """
        Initialize ensemble trainer.
        
        Args:
            ensemble_model: Ensemble model (SegmentationEnsemble, MultiScaleEnsemble, or SpecialistEnsemble)
            config: Training configuration
            device: Device to train on
        """
        self.ensemble_model = ensemble_model
        self.config = config
        self.device = device
        
        # Determine ensemble type
        if isinstance(ensemble_model, SegmentationEnsemble):
            self.ensemble_type = "basic"
        elif isinstance(ensemble_model, MultiScaleEnsemble):
            self.ensemble_type = "multi_scale"
        elif isinstance(ensemble_model, SpecialistEnsemble):
            self.ensemble_type = "specialist"
        else:
            raise ValueError(f"Unknown ensemble model type: {type(ensemble_model)}")
        
        # Setup optimizers based on ensemble type
        self._setup_optimizers()
    
    def _setup_optimizers(self):
        """Setup optimizers based on ensemble type and configuration."""
        # Get optimizer parameters
        lr = self.config.get('optimizer', {}).get('lr', 0.001)
        weight_decay = self.config.get('optimizer', {}).get('weight_decay', 0.0001)
        
        if self.ensemble_type == "basic":
            # For basic ensemble, we can either:
            # 1. Train all models together
            # 2. Use pre-trained models and only fine-tune ensemble weights
            
            # Check if we should freeze base models
            freeze_base_models = self.config.get('ensemble', {}).get('freeze_base_models', False)
            
            if freeze_base_models:
                # Freeze base model parameters
                for model in self.ensemble_model.models:
                    for param in model.parameters():
                        param.requires_grad = False
                
                # Only optimize ensemble weights
                self.optimizer = optim.Adam([self.ensemble_model.weights], lr=lr, weight_decay=weight_decay)
            else:
                # Optimize all parameters
                self.optimizer = optim.Adam(self.ensemble_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        elif self.ensemble_type == "multi_scale":
            # For multi-scale ensemble, we typically train the base model (same model at different scales)
            self.optimizer = optim.Adam(self.ensemble_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        elif self.ensemble_type == "specialist":
            # For specialist ensemble, we can train each specialist model separately for their classes
            # or train the whole ensemble together
            
            # Check if we should train specialists separately
            train_specialists_separately = self.config.get('ensemble', {}).get('train_specialists_separately', False)
            
            if train_specialists_separately:
                # Create an optimizer for each specialist model
                self.optimizers = []
                for model in self.ensemble_model.models:
                    self.optimizers.append(optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))
            else:
                # Train all models together
                self.optimizer = optim.Adam(self.ensemble_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def train_step(self, batch: Dict, criterion: nn.Module) -> Tuple[torch.Tensor, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing batch data
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Get data from batch
        points = batch['points'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass - depends on ensemble type
        if self.ensemble_type == "specialist" and hasattr(self, 'optimizers'):
            # Train each specialist separately on its assigned classes
            total_loss = 0.0
            total_correct = 0
            total_points = 0
            
            for i, model in enumerate(self.ensemble_model.models):
                # Get classes assigned to this specialist
                class_assignments = self.ensemble_model.class_assignments[i]
                
                # Create mask for points with these classes
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for cls in class_assignments:
                    mask = mask | (labels == cls)
                
                # Skip if no points for this specialist
                if not torch.any(mask):
                    continue
                
                # Extract relevant points and labels
                specialist_points = points[mask].unsqueeze(0)  # Add batch dimension
                specialist_labels = labels[mask].unsqueeze(0)  # Add batch dimension
                
                # Zero gradients for this specialist
                self.optimizers[i].zero_grad()
                
                # Forward pass
                outputs = model(specialist_points)
                
                # Calculate loss
                loss = criterion(outputs, specialist_labels)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizers[i].step()
                
                # Calculate accuracy
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == specialist_labels).sum().item()
                
                # Update totals
                total_loss += loss.item() * specialist_points.size(0)
                total_correct += correct
                total_points += specialist_points.size(0) * specialist_points.size(1)
            
            return total_loss, total_correct / total_points if total_points > 0 else 0.0
            
        else:
            # Standard training for other ensemble types
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.ensemble_model(points)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.numel()
            accuracy = correct / total
            
            return loss.item(), accuracy
    
    def validate(self, dataloader: torch.utils.data.DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the ensemble model.
        
        Args:
            dataloader: Validation dataloader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.ensemble_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Get data from batch
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.ensemble_model(points)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == labels).sum().item()
                
                # Update totals
                total_loss += loss.item() * points.size(0)
                total_correct += correct
                total_samples += points.size(0) * points.size(1)
        
        # Calculate averages
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, path: str, epoch: int, best_val_loss: float):
        """
        Save checkpoint with ensemble-specific information.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            best_val_loss: Best validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'ensemble_type': self.ensemble_type,
            'model_state_dict': self.ensemble_model.state_dict(),
            'best_val_loss': best_val_loss
        }
        
        # Add optimizer state(s)
        if hasattr(self, 'optimizers'):
            optimizer_states = []
            for opt in self.optimizers:
                optimizer_states.append(opt.state_dict())
            checkpoint['optimizer_states'] = optimizer_states
        else:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        """
        Load checkpoint with ensemble-specific handling.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Tuple of (epoch, best_val_loss)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check ensemble type
        if checkpoint.get('ensemble_type') != self.ensemble_type:
            raise ValueError(f"Checkpoint ensemble type ({checkpoint.get('ensemble_type')}) "
                           f"does not match current ensemble type ({self.ensemble_type})")
        
        # Load model state
        self.ensemble_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state(s)
        if 'optimizer_states' in checkpoint and hasattr(self, 'optimizers'):
            for i, opt_state in enumerate(checkpoint['optimizer_states']):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(opt_state)
        elif 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['best_val_loss']