// urban_point_cloud_analyzer/optimization/cuda/knn_cuda_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void knn_cuda_kernel(
    const scalar_t* __restrict__ points,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_points,
    const int dim,
    const int k) {
    
    // Get batch and point index
    int batch_idx = blockIdx.x;
    int pt_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (pt_idx >= num_points) return;
    
    // Calculate offsets for current batch
    int batch_offset = batch_idx * num_points * dim;
    int idx_offset = batch_idx * num_points * k + pt_idx * k;
    
    // Point coordinates for current point
    const scalar_t* curr_pt = points + batch_offset + pt_idx * dim;
    
    // Use shared memory to store distances
    extern __shared__ float shared_distances[];
    
    // Initialize distances to large value
    for (int i = 0; i < num_points; i++) {
        shared_distances[i] = 1e10;
    }
    
    // Calculate distances to all other points
    for (int i = 0; i < num_points; i++) {
        if (i == pt_idx) {
            shared_distances[i] = -1.0;  // Mark itself with negative distance
            continue;
        }
        
        // Calculate squared distance
        float dist_sq = 0.0;
        for (int d = 0; d < dim; d++) {
            float diff = curr_pt[d] - points[batch_offset + i * dim + d];
            dist_sq += diff * diff;
        }
        
        shared_distances[i] = dist_sq;
    }
    
    __syncthreads();
    
    // Find k nearest neighbors
    for (int j = 0; j < k; j++) {
        float min_dist = 1e10;
        int min_idx = -1;
        
        for (int i = 0; i < num_points; i++) {
            if (shared_distances[i] >= 0 && shared_distances[i] < min_dist) {
                min_dist = shared_distances[i];
                min_idx = i;
            }
        }
        
        if (min_idx == -1) {
            // Not enough points, duplicate the last valid index
            min_idx = (j > 0) ? indices[idx_offset + j - 1] : pt_idx;
        } else {
            // Mark this point as used by setting its distance to -1
            shared_distances[min_idx] = -1;
        }
        
        indices[idx_offset + j] = min_idx;
    }
}

torch::Tensor knn_cuda_forward(
    torch::Tensor points,
    int k) {
    
    // Get tensor dimensions
    auto batch_size = points.size(0);
    auto num_points = points.size(1);
    auto dim = points.size(2);
    
    // Allocate output tensor
    auto indices = torch::empty({batch_size, num_points, k}, 
                               torch::dtype(torch::kInt64).device(points.device()));
    
    // Calculate grid and block dimensions
    const int threads = 256;
    const dim3 blocks(batch_size, (num_points + threads - 1) / threads);
    
    // Shared memory size
    const int smem_size = num_points * sizeof(float);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "knn_cuda", ([&] {
        knn_cuda_kernel<scalar_t><<<blocks, threads, smem_size, at::cuda::getCurrentCUDAStream()>>>(
            points.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            num_points,
            dim,
            k);
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return indices;
}