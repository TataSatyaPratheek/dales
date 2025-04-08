#!/bin/bash

# Urban Point Cloud Analyzer - Project Setup Script
# This script creates the directory structure for the project and sets up virtual environment

# Stop on errors
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project name
PROJECT_NAME="urban_point_cloud_analyzer"

echo -e "${GREEN}Creating project structure for ${PROJECT_NAME}...${NC}"

# Create main project directory
mkdir -p ${PROJECT_NAME}
cd ${PROJECT_NAME}

# Create main module directory
mkdir -p ${PROJECT_NAME}

# Create the README.md file
touch README.md

# Create the project structure
directories=(
    "data/preprocessing" 
    "data/augmentation" 
    "data/loaders"
    "models/backbones"
    "models/segmentation"
    "models/detection"
    "models/ensemble"
    "training/loss_functions"
    "training/optimizers"
    "training/schedulers"
    "evaluation"
    "visualization"
    "inference"
    "optimization/cuda"
    "optimization/quantization"
    "business/metrics"
    "business/reports"
    "business/roi"
    "api"
    "ui"
    "utils"
    "configs"
    "tests"
    "notebooks"
    "scripts"
)

# Create directories
for dir in "${directories[@]}"; do
    mkdir -p "${PROJECT_NAME}/${dir}"
    touch "${PROJECT_NAME}/${dir}/__init__.py"
    echo -e "${GREEN}Created ${dir}${NC}"
done

# Create main __init__.py
touch ${PROJECT_NAME}/__init__.py

# Create setup.py
cat > setup.py << 'EOL'
from setuptools import setup, find_packages

setup(
    name="urban_point_cloud_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision",
        "open3d",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "wandb",
        "tensorboard",
        "laspy",
        "pyproj",
        "shapely",
        "geopandas",
        "pandas",
        "dash",
        "plotly",
    ],
    python_requires=">=3.8",
)
EOL

# Create requirements.txt
cat > requirements.txt << 'EOL'
# Core dependencies
torch>=1.9.0
torchvision
open3d
numpy
scipy
scikit-learn

# Point cloud specific
laspy
pyproj
shapely
geopandas
pandas
pointnet2_ops_lib # May need to be installed separately

# Visualization
matplotlib
plotly
dash
dash-bootstrap-components

# Training and logging
tqdm
pyyaml
wandb
tensorboard

# Testing
pytest
pytest-cov

# Development
black
isort
flake8
mypy
EOL

# Create example config file
mkdir -p configs
cat > configs/default_config.yaml << 'EOL'
# Default configuration for Urban Point Cloud Analyzer

data:
  dataset: "DALES"
  data_path: "data/DALES"
  cache_path: "data/cache"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  voxel_size: 0.05
  augmentation:
    enabled: true
    rotation_range: 15
    flip_probability: 0.5
    dropout_probability: 0.1
    noise_sigma: 0.01

model:
  type: "pointnet2_seg"  # Options: pointnet2_seg, kpconv, randlanet, spvcnn
  backbone:
    name: "pointnet2"
    use_normals: true
    num_points: 16384
    num_layers: 4
  segmentation:
    num_classes: 8  # DALES has 8 classes
    dropout: 0.5
  detection:
    enabled: true
    min_confidence: 0.5
    nms_threshold: 0.3

training:
  batch_size: 8
  epochs: 100
  optimizer:
    name: "adam"
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    name: "cosine"
    min_lr: 0.00001
  loss:
    segmentation: "cross_entropy"
    detection: "focal"
  mixed_precision: true
  grad_clip: 1.0
  save_checkpoint_steps: 1000
  validation_steps: 500
  early_stopping_patience: 10

inference:
  batch_size: 4
  sliding_window:
    enabled: true
    window_size: 20
    stride: 10
  test_time_augmentation: true

optimization:
  quantization:
    enabled: false
    precision: "int8"
  cuda:
    enabled: true
    custom_kernels: true

business:
  metrics:
    - "green_coverage"
    - "building_density"
    - "road_quality"
  reports:
    format: ["pdf", "json"]
    include_visualizations: true
  roi:
    calculation_method: "comparative"

visualization:
  colormap: "tab10"
  point_size: 2
  background_color: [0.9, 0.9, 0.9]
  class_colors:
    ground: [0.95, 0.95, 0.95]
    vegetation: [0.0, 0.8, 0.0]
    building: [0.7, 0.7, 0.7]
    water: [0.0, 0.0, 0.8]
    car: [1.0, 0.0, 0.0]
    truck: [0.8, 0.4, 0.0]
    powerline: [1.0, 1.0, 0.0]
    fence: [0.6, 0.3, 0.0]
EOL

# Create example entry point script
mkdir -p scripts
cat > scripts/train.py << 'EOL'
#!/usr/bin/env python3
"""
Training script for Urban Point Cloud Analyzer
"""
import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urban_point_cloud_analyzer.data.loaders import DALESDataset
from urban_point_cloud_analyzer.models import get_model
from urban_point_cloud_analyzer.training import get_loss_function, get_optimizer, get_scheduler
from urban_point_cloud_analyzer.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Urban Point Cloud Analyzer")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_path'] = args.data_dir
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logger
    logger = setup_logger(output_dir / "train.log")
    logger.info(f"Config: {config}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    logger.info("Creating datasets...")
    # TODO: Implement actual dataset and dataloader
    
    # Create model
    logger.info("Creating model...")
    # TODO: Implement model creation
    
    # Define loss function, optimizer and scheduler
    logger.info("Setting up training...")
    # TODO: Implement training setup
    
    # Training loop
    logger.info("Starting training...")
    # TODO: Implement training loop
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
EOL

# Create a Docker file
cat > Dockerfile << 'EOL'
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["bash"]
EOL

# Create a simple gitignore file
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Data
data/DALES/
data/cache/
*.h5
*.las
*.laz
*.ply
*.pcd

# Models
outputs/
runs/
checkpoints/
*.pth
*.pt
*.bin

# Logs
logs/
*.log
wandb/
lightning_logs/

# OS specific
.DS_Store
Thumbs.db
EOL

# Create a README.md file
echo "# Urban Point Cloud Analyzer" > README.md
echo "Comprehensive solution for 3D point cloud analysis in urban planning" >> README.md

# Create a virtual environment if Python is available
if command -v python3 &>/dev/null; then
    echo -e "${YELLOW}Setting up virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created. Activate it with:${NC}"
    echo -e "${YELLOW}source venv/bin/activate${NC}"
else
    echo -e "${YELLOW}Python3 not found. Virtual environment not created.${NC}"
fi

echo -e "${GREEN}Project structure created successfully!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Activate virtual environment: ${GREEN}source venv/bin/activate${NC}"
echo -e "2. Install dependencies: ${GREEN}pip install -r requirements.txt${NC}"
echo -e "3. Install package in development mode: ${GREEN}pip install -e .${NC}"
echo -e "4. Set up your DALES dataset: ${GREEN}python scripts/prepare_dales.py --data_dir /path/to/dalesObject.tar.gz${NC}"