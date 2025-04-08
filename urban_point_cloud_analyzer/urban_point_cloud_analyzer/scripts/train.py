# urban_point_cloud_analyzer/scripts/train.py
#!/usr/bin/env python3
"""
Training script for Urban Point Cloud Analyzer
"""
import argparse
import os
import sys
import yaml
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.data.loaders import DALESDataset
from urban_point_cloud_analyzer.data.augmentation import PointCloudTransforms
from urban_point_cloud_analyzer.data.preprocessing import PointCloudPreprocessor
from urban_point_cloud_analyzer.models import get_model
from urban_point_cloud_analyzer.training.loss_functions import get_loss_function
from urban_point_cloud_analyzer.training.optimizers import get_optimizer
from urban_point_cloud_analyzer.training.schedulers import get_scheduler
from urban_point_cloud_analyzer.utils.logger import setup_logger
from urban_point_cloud_analyzer.utils.hardware_utils import get_device_info, get_optimal_config, optimize_model_for_device, optimize_dataloader_for_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train Urban Point Cloud Analyzer")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA even if available")
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Set up progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    # Get hardware info
    device_info = get_device_info()
    
    # Mixed precision training
    use_amp = config.get('mixed_precision', False) and device_info.get('cuda_available', False)
    scaler = GradScaler() if use_amp else None
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        points = batch['points'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast():
                outputs = model(points)
                loss = criterion(outputs, labels)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Apply gradient clipping
            if config.get('grad_clip', 0.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip'))
            
            # Update weights with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping
            if config.get('grad_clip', 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip'))
            
            # Update weights
            optimizer.step()
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        
        # Update statistics
        total_loss += loss.item() * points.size(0)
        total_correct += correct
        total_samples += points.size(0) * points.size(1)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': correct / (points.size(0) * points.size(1))
        })
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = total_correct / total_samples
    
    # Log metrics
    logger.info(f"Epoch {epoch} [Train] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, logger):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Set up progress bar
    pbar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Get data
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            
            # Update statistics
            total_loss += loss.item() * points.size(0)
            total_correct += correct
            total_samples += points.size(0) * points.size(1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / (points.size(0) * points.size(1))
            })
    
    # Calculate epoch metrics
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = total_correct / total_samples
    
    # Log metrics
    logger.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    return val_loss, val_acc


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_path'] = args.data_dir
    
    # Set timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config for reproducibility
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Setup logger
    logger = setup_logger(output_dir / "train.log")
    logger.info(f"Config: {config}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get hardware info and optimal config
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Override CUDA availability if requested
    if args.disable_cuda:
        device_info['cuda_available'] = False
    
    # Get optimal config for hardware
    optimal_config = get_optimal_config(device_info)
    logger.info(f"Optimal config: {optimal_config}")
    
    # Update config with hardware-specific settings
    config['training']['batch_size'] = optimal_config['train_batch_size']
    config['training']['mixed_precision'] = optimal_config['mixed_precision']
    
    # Setup device
    device = torch.device("cuda" if device_info['cuda_available'] else 
                         "mps" if device_info.get('mps_available', False) else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    logger.info("Creating datasets...")
    
    # Setup transformations
    transforms = PointCloudTransforms(config['data']['augmentation'])
    
    # Setup preprocessor
    preprocessor = PointCloudPreprocessor(config['data'])
    
    # Create train dataset
    train_dataset = DALESDataset(
        root_dir=config['data']['data_path'],
        split='train',
        config=config['data'],
        transforms=transforms,
        preprocessor=preprocessor,
        cache=True,
        num_points=config['model']['backbone'].get('num_points', 16384),
        memory_efficient=device_info.get('is_m1', False)  # Use memory-efficient mode for M1
    )
    
    # Create validation dataset
    val_dataset = DALESDataset(
        root_dir=config['data']['data_path'],
        split='val',
        config=config['data'],
        transforms=None,  # No augmentation for validation
        preprocessor=preprocessor,
        cache=True,
        num_points=config['model']['backbone'].get('num_points', 16384),
        memory_efficient=device_info.get('is_m1', False)  # Use memory-efficient mode for M1
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Optimize dataloader configuration
    train_loader_config = {
        'batch_size': config['training']['batch_size'],
        'shuffle': True,
        'drop_last': True,
        'collate_fn': DALESDataset.collate_fn,
        'split': 'train'
    }
    
    val_loader_config = {
        'batch_size': config['training']['batch_size'] * 2,  # Larger batch for validation
        'shuffle': False,
        'drop_last': False,
        'collate_fn': DALESDataset.collate_fn,
        'split': 'val'
    }
    
    train_loader_config = optimize_dataloader_for_device(train_loader_config, device_info)
    val_loader_config = optimize_dataloader_for_device(val_loader_config, device_info)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, **train_loader_config)
    val_loader = DataLoader(val_dataset, **val_loader_config)
    
    logger.info(f"Train loader: {train_loader_config}")
    logger.info(f"Validation loader: {val_loader_config}")
    
    # Create model
    logger.info("Creating model...")
    model = get_model(config['model'])
    
    # Log model architecture
    logger.info(f"Model: {model}")
    
    # Optimize model for device
    model = optimize_model_for_device(model, device_info)
    
    # Setup loss function
    criterion = get_loss_function(config['training']['loss'])
    
    # Setup optimizer
    optimizer = get_optimizer(model.parameters(), config['training']['optimizer'])
    
    # Setup scheduler
    scheduler = get_scheduler(optimizer, config['training']['scheduler'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    
    num_epochs = config['training']['epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, config['training']
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, logger)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_val_loss': best_val_loss
        }, output_dir / f"checkpoint_epoch_{epoch}.pth")
        
        # Save best model
        if is_best:
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info("Training completed!")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pth")


if __name__ == "__main__":
    main()