#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <glog/logging.h>
#include <stdio.h>
#include "./util.h"


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

__device__ void thread_reduce(TD* data, TD* shared_data, TI* shared_index, TD lowest, int bsize, int hsize) { 
  //if (threadIdx.x > hsize) return;
  if (blockIdx.x > bsize) return;

  TD * batch_data = data + blockIdx.x * hsize;
  TD  thread_local_max_data = lowest;
  TI  thread_local_max_idx = 0;

  for (int i = threadIdx.x ; i < hsize; i += blockDim.x) {
    if (batch_data[i] > thread_local_max_data) { 
      thread_local_max_data = batch_data[i];
      thread_local_max_idx = i;
    }
  }
  
  shared_data[threadIdx.x] = thread_local_max_data;
  shared_index[threadIdx.x] = thread_local_max_idx;
}


__device__ void block_reduce(TD* shared_data, TI* shared_index) {
  int left, right;
  int threads = blockDim.x / 2;

  for (int stride = 1; stride < blockDim.x; stride *= 2, threads /=2 ) {
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

__global__ void argmax_kernel(TD* data, TI* index_out, TD lowest, int bsize, int hsize) { 
  // shared memory  
  extern __shared__ int s[]; 
  TD* shared_data = reinterpret_cast<TD*>(s);
  TI* shared_index = reinterpret_cast<TI*>(shared_data + blockDim.x);

  thread_reduce(data, shared_data, shared_index, lowest, bsize, hsize);
  __syncthreads();
  // for each block do block reduce
  block_reduce(shared_data, shared_index);

  // write to global memory
  if (threadIdx.x == 0) {
    index_out[blockIdx.x] = shared_index[0];
  }
}


int main(int argc, char **argv) {
  int bsize = 64;
  int hsize = 30000;

  TD* data_host = (TD*)malloc(sizeof(TD) * hsize * bsize);
  TI* index_host = (TI*)malloc(sizeof(TI) * bsize);
  
  for (int i = 0;i < bsize;i ++) { 
    for (int j = 0;j < hsize;j ++) { 
      data_host[j + i * hsize] = (TD)j;
    }
  }

  LOG(INFO) << " host set up ok.";
  // copy to cuda
  TD * data_dev;
  TI * index_dev;
  CUCHECK(cudaMalloc(reinterpret_cast<void**>(&data_dev), sizeof(TD) * hsize * bsize));
  CUCHECK(cudaMalloc(reinterpret_cast<void**>(&index_dev), sizeof(TI) * bsize));

  CUCHECK(cudaMemcpy(data_dev, data_host, sizeof(TD) * hsize * bsize, cudaMemcpyHostToDevice));
  //CUCHECK(cudaMemcpy(index_dev, index_host, sizeof(TD) * hsize * bsize), cudaMemcpyHostToDevice);

  LOG(INFO) << " dev set up ok.";
  // launch kernel
  int numBlocks = bsize;
  int threadsPerBlock = 512;
  size_t dev_sm_bytes = threadsPerBlock * (sizeof(TD) + sizeof(TI));

  for (int i = 0;i < 10;i ++) {
    uint64_t t0, t1;
    t0 = get_time();
    argmax_kernel<<<numBlocks, threadsPerBlock, dev_sm_bytes>>>(data_dev, index_dev, 
      std::numeric_limits<TD>::lowest(), bsize, hsize);
    CUCHECK(cudaDeviceSynchronize());
    t1 = get_time();
    LOG(INFO) << " takes us : " << t1 - t0;
  }

  LOG(INFO) << " argmax ok.";
  // get output and print
  CUCHECK(cudaMemcpy(index_host, index_dev, sizeof(TI) * bsize, cudaMemcpyDeviceToHost));
  
  for (int i = 0;i < bsize;i ++) { 
    LOG(INFO) << " " << index_host[i];
  }

  return 0;
}







