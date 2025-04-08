// urban_point_cloud_analyzer/optimization/cuda/knn_cuda_optimized.cpp
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor knn_cuda_optimized_forward(
    torch::Tensor points,
    int k);

// C++ interface
torch::Tensor knn_optimized_forward(
    torch::Tensor points,
    int k) {
  return knn_cuda_optimized_forward(points, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn_optimized", &knn_optimized_forward, "Optimized K-Nearest Neighbors (CUDA)");
}