// urban_point_cloud_analyzer/optimization/cuda/knn_cuda_optimized.cu
// Optimized KNN CUDA kernel for NVIDIA 1650Ti
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// Optimized for 1650Ti with better memory coalescing and shared memory usage
template <typename scalar_t>
__global__ void knn_cuda_optimized_kernel(
    const scalar_t* __restrict__ points,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_points,
    const int dim,
    const int k) {
    
    // Get batch and point index
    const int batch_idx = blockIdx.x;
    const int pt_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (pt_idx >= num_points) return;
    
    // Calculate offsets for current batch
    const int batch_offset = batch_idx * num_points * dim;
    const int idx_offset = batch_idx * num_points * k + pt_idx * k;
    
    // Point coordinates for current point
    const scalar_t* curr_pt = points + batch_offset + pt_idx * dim;
    
    // Use shared memory to store coordinates of current point
    extern __shared__ char shared_mem[];
    scalar_t* shared_curr_pt = (scalar_t*)shared_mem;
    
    // Each thread loads one component of the current point
    if (threadIdx.x < dim) {
        shared_curr_pt[threadIdx.x] = curr_pt[threadIdx.x];
    }
    __syncthreads();
    
    // Calculate distances to all other points
    scalar_t* distances = (scalar_t*)(shared_mem + dim * sizeof(scalar_t));
    for (int i = 0; i < num_points; i++) {
        if (i == pt_idx) {
            distances[threadIdx.x] = -1.0;  // Mark itself with negative distance
            continue;
        }
        
        // Calculate squared distance
        scalar_t dist_sq = 0.0;
        for (int d = 0; d < dim; d++) {
            scalar_t diff = shared_curr_pt[d] - points[batch_offset + i * dim + d];
            dist_sq += diff * diff;
        }
        
        distances[threadIdx.x] = dist_sq;
    }
    
    __syncthreads();
    
    // Find k smallest distances using a heap (more efficient for 1650Ti)
    // This reduces register pressure and thread divergence
    typedef struct {
        scalar_t dist;
        int idx;
    } DistIdx;
    
    // Initialize a max heap in registers (better for 1650Ti)
    DistIdx heap[32];  // Use a fixed size that fits in registers (k is typically small)
    const int actual_k = min(k, 32);  // Cap at 32 for register efficiency
    
    // Initialize with large values
    for (int j = 0; j < actual_k; j++) {
        heap[j].dist = 1e10;
        heap[j].idx = -1;
    }
    
    // Process each point and maintain a max heap of smallest distances
    for (int i = 0; i < num_points; i++) {
        scalar_t dist = distances[i];
        if (dist >= 0 && dist < heap[0].dist) {
            // Replace max and sift down
            heap[0].dist = dist;
            heap[0].idx = i;
            
            // Sift down
            int parent = 0;
            while (true) {
                int child1 = 2 * parent + 1;
                int child2 = 2 * parent + 2;
                
                if (child1 >= actual_k) break;
                
                int max_child = (child2 < actual_k && heap[child2].dist > heap[child1].dist) ? child2 : child1;
                
                if (heap[parent].dist >= heap[max_child].dist) break;
                
                // Swap
                DistIdx temp = heap[parent];
                heap[parent] = heap[max_child];
                heap[max_child] = temp;
                
                parent = max_child;
            }
        }
    }
    
    // Sort the heap (insertion sort is efficient for small k)
    for (int i = 1; i < actual_k; i++) {
        DistIdx key = heap[i];
        int j = i - 1;
        
        while (j >= 0 && heap[j].dist > key.dist) {
            heap[j + 1] = heap[j];
            j--;
        }
        
        heap[j + 1] = key;
    }
    
    // Copy results to output
    for (int j = 0; j < actual_k; j++) {
        indices[idx_offset + j] = heap[j].idx >= 0 ? heap[j].idx : pt_idx;
    }
    
    // Fill remaining indices if k > 32
    for (int j = actual_k; j < k; j++) {
        indices[idx_offset + j] = indices[idx_offset + actual_k - 1];
    }
}

torch::Tensor knn_cuda_optimized_forward(
    torch::Tensor points,
    int k) {
    
    // Get tensor dimensions
    auto batch_size = points.size(0);
    auto num_points = points.size(1);
    auto dim = points.size(2);
    
    // Allocate output tensor
    auto indices = torch::empty({batch_size, num_points, k}, 
                               torch::dtype(torch::kInt64).device(points.device()));
    
    // Calculate grid and block dimensions optimized for 1650Ti
    // 1650Ti has 16 SMs, each with 128 CUDA cores
    const int threads = 128;  // Match the warp size 
    const dim3 blocks(batch_size, (num_points + threads - 1) / threads);
    
    // Shared memory size for current point coordinates and distances
    const int smem_size = sizeof(float) * (dim + threads);
    
    // Launch kernel with stream from current CUDA context
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "knn_cuda_optimized", ([&] {
        knn_cuda_optimized_kernel<scalar_t><<<blocks, threads, smem_size, at::cuda::getCurrentCUDAStream()>>>(
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