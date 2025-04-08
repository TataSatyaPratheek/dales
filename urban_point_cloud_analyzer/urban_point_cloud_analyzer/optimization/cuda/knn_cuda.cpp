// urban_point_cloud_analyzer/optimization/cuda/knn_cuda.cpp
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor knn_cuda_forward(
    torch::Tensor points,
    int k);

// C++ interface
torch::Tensor knn_forward(
    torch::Tensor points,
    int k) {
  return knn_cuda_forward(points, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn_forward, "K-Nearest Neighbors (CUDA)");
}