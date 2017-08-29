#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <glog/logging.h>


#define CUCHECK(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      //fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
      LOG(FATAL) <<  "CUDA Error: "<< cudaGetErrorString(stat) << " "
                 << file << " "
                 << line;
   }
}


using TD = float;
using TI = int;


/*
 * Each thread do reduction, and write to global shared memory.
 */

__device__ thread_reduce(TD* data, TD* data_out, TI* index_out, int bsize, int hsize) { 
  //if (threadIdx.x > hsize) return;
  if (blockIdx.x > bsize) return;

  TD * batch_data = data + blockIdx.x * hsize;
  //TD  thread_local_max_data = batch_data[threadIdx.x];
  //TI  thread_local_max_idx = threadIdx.x;
  TD  thread_local_max_data = std::numeric_limits<TD>::lowest();
  TI  thread_local_max_idx = 0;

  for (int i = threadIdx.x ; i < hsize; i += blockDim.x) {
    if (batch_data[i] > thread_local_max_data) { 
      thread_local_max_data = batch_data[i];
      thread_local_max_idx = i;
    }
  }
  data_out[blockIdx.x][threadIdx.x] = thread_local_max_data;
  index_out[blockIdx.x][threadIdx.x] = thread_local_max_idx;
}

__device__ block_reduce(TD* shared_data, TI* shared_index) {
  TD left, right;
  int threads = blockDim.x / 2;

  for (stride = 1; stride < blockDim.x; stride *= 2, threads /=2 ) {
    if (threadIdx.x < threads) {
      left = threadIdx.x * (stride * 2);
      right = left + stride;
      if (shared_data[left] < shared_data[right]) {
        shared_data[left] = shared_data[right];
        shared_index[left] = shared_index[right];
      }
    }
    __syncthreads();
  }
}

__global__ argmax_kernel(TD* data, TI* index_out, int bsize, int hsize) { 
  // shared memory  
  __shared__ TD shared_data[gridDim.x][blockDim.x];
  __shared__ TI shared_index[gridDim.x][blockDim.x];

  thread_reduce(data, shared_data, shared_index, bsize, hsize);
  __syncthreads();
  // for each block do block reduce
  block_reduce(shared_data, shared_index);

  // write to global memory
  if (threadIdx.x < bsize) { 
    index_out[threadIdx.x] = shared_index[blockIdx.x][threadIdx.x];
  }
}


int main(int argc, char **argv) {
  int bsize = 1;
  int hsize = 30000;

  TD* data_host = malloc(sizeof(TD) * hsize * bsize);
  TI* index_host = malloc(sizeof(TI) * bsize);
  
  for (int i = 0;i < bsize;i ++) { 
    for (int j = 0;j < hsize;j ++) { 
      data_host[j + i * hsize] = (TD)j;
    }
  }

  // copy to cuda
  TD * data_dev;
  TI * index_dev;
  CUCHECK(cudaMalloc(static_cast<void**>(&data_dev), sizeof(TD) * hsize * bsize));
  CUCHECK(cudaMalloc(static_cast<void**>(&index_dev), sizeof(TI) * bsize));

  CUCHECK(cudaMemcpy(data_dev, data_host, sizeof(TD) * hsize * bsize), cudaMemcpyHostToDevice);
  //CUCHECK(cudaMemcpy(index_dev, index_host, sizeof(TD) * hsize * bsize), cudaMemcpyHostToDevice);

  // launch kernel
  int numBlocks = bsize;
  int threadsPerBlock = 512;

  argmax_kernel<<<numBlocks, threadsPerBlock>>>(data_dev, index_dev);

  CUCHECK(cudaDeviceSynchronize());

  // get output and print
  CUCHECK(cudaMemcpy(index_host, index_dev, sizeof(TI) * bsize), cudaMemcpyDeviceToHost);
  
  for (int i = 0;i < bsize;i ++) { 
    LOG(INFO) << " " << index_host[i];
  }


  return 0;
}







