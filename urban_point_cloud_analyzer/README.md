# Urban Point Cloud Analyzer

![Urban Point Cloud Analysis](https://img.shields.io/badge/Urban-Point%20Cloud%20Analysis-blue)
![DALES Dataset](https://img.shields.io/badge/Dataset-DALES-green)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![CUDA Optimized](https://img.shields.io/badge/CUDA-Optimized-red)
![M1 Compatible](https://img.shields.io/badge/M1-Compatible-purple)

## Comprehensive 3D Point Cloud Analysis for Urban Planning Applications

The Urban Point Cloud Analyzer is a production-ready deep learning solution that processes LiDAR point cloud data to support data-driven urban planning decisions. Built with PyTorch and optimized for multiple hardware platforms, it provides both technical excellence and actionable business intelligence.

## 🏙️ Overview

This project demonstrates advanced expertise in point cloud deep learning for LiDAR data, with a focus on commercial applications in urban planning. Using the DALES (DAta fusion contest LiDAR classification of urban Environment for Semantic segmentation) dataset, this solution offers semantic segmentation, object detection, and business intelligence to solve real-world urban planning challenges.

### 🔑 Key Features

- **3D Semantic Segmentation**: Classify urban elements (buildings, vegetation, roads, etc.)
- **Object Detection**: Identify and count urban infrastructure components
- **Urban Metrics Calculation**: Generate quantitative measurements of urban characteristics
- **Business Intelligence**: Translate technical findings into actionable business insights
- **Performance Optimization**: Hardware-specific optimizations for multiple targets:
  - CUDA optimization for NVIDIA GPUs (specifically 1650Ti)
  - Metal acceleration for Apple M1 Macs
  - Memory-efficient operations for resource-constrained devices
- **Interactive Visualization**: Explore and present findings through intuitive interfaces

## 🛠️ Technical Implementation

### Hardware-Specific Optimizations

The project includes comprehensive optimizations for different hardware platforms:

#### NVIDIA GTX 1650Ti (4GB VRAM)
- **Mixed Precision Training**: Reduces memory usage by using fp16 for appropriate operations
- **Sparse Tensor Operations**: Optimized data structures for point clouds
- **Gradient Checkpointing**: Trades computation for memory to fit larger models
- **Batch Size Optimization**: Automatically determines optimal batch size
- **Custom CUDA Kernels**: Highly optimized kNN implementation

#### Apple M1 MacBook Air (8GB RAM)
- **Metal Performance Shaders (MPS)**: Acceleration on M1's Neural Engine
- **Memory-Efficient Point Cloud Processing**: Automatic chunking for large point clouds
- **Model Quantization**: Float16 and Int8 precision for faster inference
- **Unified Memory Optimizations**: Respects the shared memory architecture

#### General Optimizations
- **Automatic Hardware Detection**: Applies appropriate optimizations based on hardware
- **Dynamic Batch Size Selection**: Maximizes throughput within memory constraints
- **Sparse Convolution**: Efficient convolution for sparse 3D data
- **Modular Architecture**: Clean separation of concerns for maintainability

### Architecture

The project follows a modular architecture with clear separation of concerns:

```
urban_point_cloud_analyzer/
├── data/                      # Data handling
│   ├── preprocessing/         # Preprocessing pipelines
│   ├── augmentation/          # Data augmentation
│   └── loaders/               # Dataset loaders
├── models/                    # Model architectures
│   ├── backbones/             # Feature extraction networks
│   ├── segmentation/          # Semantic segmentation
│   ├── detection/             # Object detection
│   └── ensemble/              # Model ensemble strategies
├── training/                  # Training pipelines
│   ├── loss_functions/        # Custom loss functions
│   ├── optimizers/            # Optimizer configurations
│   └── schedulers/            # Learning rate schedulers
├── evaluation/                # Evaluation metrics
├── visualization/             # Visualization tools
├── inference/                 # Inference pipelines
├── optimization/              # Performance optimization
│   ├── cuda/                  # CUDA kernels
│   ├── mixed_precision.py     # Mixed precision training
│   ├── sparse_ops.py          # Sparse tensor operations
│   ├── gradient_checkpointing.py # Memory-efficient training
│   ├── batch_size_optimization.py # Automatic batch sizing
│   ├── m1_optimizations.py    # Apple M1 optimizations
│   ├── hardware_optimizations.py # Hardware detection & optimization
│   └── quantization/          # Model quantization
├── business/                  # Business intelligence
│   ├── metrics/               # Urban metrics calculation
│   ├── reports/               # Report generation
│   └── roi/                   # ROI calculation
├── api/                       # API endpoints
├── ui/                        # User interface
├── tests/                     # Test suite
└── utils/                     # Utility functions
```

### Model Architecture Options

The framework supports multiple state-of-the-art point cloud processing architectures:

1. **Primary Models**:
   - **PointNet++**: Pioneer in direct point cloud processing
   - **KPConv**: Kernel point convolution for irregular point clouds
   - **RandLA-Net**: Efficient large-scale processing with random sampling
   - **SPVCNN**: Sparse voxel-based convolution

2. **Ensemble Strategy**:
   - Multi-scale feature fusion
   - Model averaging with confidence weighting
   - Specialist models for different urban elements

### Business Intelligence Components

Transform technical results into actionable insights:

1. **Urban Metrics Calculator**:
   - Building density calculation
   - Green space coverage analysis
   - Road network connectivity measurements
   - Accessibility scoring

2. **Decision Support System**:
   - Impact assessment of proposed changes
   - Resource allocation optimization
   - Compliance checking against regulations

3. **ROI Calculator**:
   - Maintenance cost estimation
   - Environmental impact quantification
   - Urban planning benefit analysis

4. **Reporting System**:
   - Interactive dashboards
   - Automated report generation
   - Comparative analysis over time

## 💾 Dataset: DALES

The DALES dataset contains aerial LiDAR point clouds of urban environments with the following characteristics:

- **Size**: ~10 billion points across 40 scenes
- **Resolution**: ~10 points/m²
- **Classes**: 8 semantic classes (ground, vegetation, buildings, etc.)
- **Format**: LAS/LAZ files with XYZ coordinates, intensity, and class labels
- **Coverage**: Urban and suburban areas with varied structures

### DALES Classes

| Class ID | Class Name | Description | Color |
|----------|------------|-------------|-------|
| 0 | Ground | Terrain and bare earth | Gray |
| 1 | Vegetation | Trees, bushes, and other plants | Green |
| 2 | Buildings | Structures and constructions | Brown |
| 3 | Water | Rivers, lakes, and ponds | Blue |
| 4 | Car | Automobiles and personal vehicles | Red |
| 5 | Truck | Large vehicles and transport | Orange |
| 6 | Powerline | Cables and transmission lines | Yellow |
| 7 | Fence | Barriers and boundaries | Purple |

## 📊 Evaluation Metrics

### Technical Metrics

- **Mean Intersection over Union (mIoU)**: Measures segmentation accuracy
- **Average Precision (AP)**: Measures object detection accuracy
- **Inference Time**: Processing speed on target hardware
- **Memory Usage**: RAM and VRAM consumption

### Business Metrics

- **Urban Analysis Accuracy**: Comparison with ground truth measurements
- **Decision Support Quality**: Correlation of recommendations with expert decisions
- **Time Savings**: Comparison with manual analysis methods
- **Return on Investment**: Cost reduction estimations for urban planning

## 🔧 Hardware Requirements

### Development Hardware

- **Primary Development**: Ryzen 4800H with NVIDIA 1650Ti

### Deployment Targets

- **Target 1**: Desktop/Server with CUDA GPU
- **Target 2**: Apple M1 MacBook Air (8GB RAM)

### Software Requirements

- **Python**: 3.8+
- **Framework**: PyTorch 1.9+ with CUDA support
- **Key Libraries**:
  - `open3d`: Point cloud processing
  - `laspy`: LAS/LAZ file handling
  - `pyproj`: Coordinate system operations
  - `scikit-learn`: Machine learning utilities
  - `dash`/`plotly`: Interactive visualization
  - `pointnet2_ops_lib`: PointNet++ operations

## 🚀 Getting Started

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/urban_point_cloud_analyzer.git
   cd urban_point_cloud_analyzer
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode
   ```bash
   pip install -e .
   ```

5. Install PointNet++ operations
   ```bash
   pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
   ```

6. Prepare the DALES dataset
   ```bash
   python scripts/prepare_dales.py --data_dir /path/to/dalesObject.tar.gz
   ```

### Basic Usage

1. Train a model with automatic hardware optimization
   ```bash
   python scripts/train.py --config configs/default_config.yaml
   ```

2. Run inference
   ```bash
   python scripts/inference.py --model_path outputs/best_model.pth --data_path path/to/point_cloud.las
   ```

3. Generate urban metrics
   ```bash
   python scripts/generate_metrics.py --model_path outputs/best_model.pth --data_path path/to/point_cloud.las
   ```

4. Launch visualization dashboard
   ```bash
   python scripts/dashboard.py --results_dir outputs/results
   ```

## 💻 Code Examples

### Hardware-Optimized Training

```python
from urban_point_cloud_analyzer.optimization import get_optimization_manager
from urban_point_cloud_analyzer.models import get_model
from urban_point_cloud_analyzer.data.loaders import DALESDataset

# Create model
model = get_model(config['model'])

# Create optimization manager and optimize model for current hardware
opt_manager = get_optimization_manager()
model = opt_manager.optimize_model(model)

# Create trainer with appropriate optimizations
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = opt_manager.get_trainer(model, optimizer)

# Get optimal batch size for current hardware
batch_size = opt_manager.get_optimal_batch_size(model, (3, 32, 32))
print(f"Using optimal batch size: {batch_size}")

# Create dataset and dataloader
dataset = DALESDataset(root_dir="data/DALES", split="train")
dataloader_config = {'batch_size': batch_size, 'shuffle': True}
dataloader = torch.utils.data.DataLoader(
    dataset, 
    **opt_manager.optimize_dataloader_config(dataloader_config)
)

# Train with optimized settings
for epoch in range(10):
    for batch in dataloader:
        loss, acc = trainer.train_step(batch['points'], batch['labels'], cross_entropy_loss)
        print(f"Epoch {epoch}, Loss: {loss.item()}, Acc: {acc}")
```

### Urban Analysis

```python
from urban_point_cloud_analyzer.business.metrics.integrated_metrics import IntegratedUrbanAnalyzer
import laspy
import numpy as np

# Load point cloud with segmentation
las = laspy.read("segmented_point_cloud.las")
points = np.vstack([las.x, las.y, las.z]).T
labels = np.array(las.classification)

# Create urban analyzer
analyzer = IntegratedUrbanAnalyzer()

# Calculate comprehensive urban metrics
metrics = analyzer.analyze(points, labels)

# Generate human-readable report
report = analyzer.generate_report(metrics)
print(report)

# Access specific metrics
print(f"Building density: {metrics['building_density']:.2f}")
print(f"Green coverage: {metrics['green_coverage']:.2f}")
print(f"Urban quality score: {metrics['urban_quality_score']:.2f}/100")
```

## 🤝 Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

1. Varney, N., Asari, V.K., Graehling, Q. (2020). DALES: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.

2. Qi, C.R., Yi, L., Su, H., Guibas, L.J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. In: Advances in Neural Information Processing Systems (NIPS).

3. Thomas, H., Qi, C.R., Deschaud, J.E., Marcotegui, B., Goulette, F., Guibas, L.J. (2019). KPConv: Flexible and Deformable Convolution for Point Clouds. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

4. Hu, Q., Yang, B., Xie, L., Rosa, S., Guo, Y., Wang, Z., Trigoni, N., Markham, A. (2020). RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

5. Tang, H., Liu, Z., Zhao, S., Lin, Y., Lin, J., Wang, H., Han, S. (2020). Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution. In: European Conference on Computer Vision (ECCV).

use pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
