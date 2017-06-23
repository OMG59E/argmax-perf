
#define CUB_STDERR
#include <iostream>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/util_debug.cuh>
#include "util.h"

#define T float

using namespace cub;

using KVT = KeyValuePair<int, T>;

void init(T *h_in, int num_item, int stride){
  for(int i = 0;i < num_item;i ++) {
    h_in[i] = i % stride;
  }
}

KVT* get_out(KVT* d_data, size_t num) {
  KVT* ret = (KVT*)malloc(sizeof(KVT) * num);
  CubDebugExit(cudaMemcpy(ret, d_data, sizeof(KVT) * num, cudaMemcpyDeviceToHost));
  return ret;
}

int main(int argc, char **argv) {

  int bsize = 32;
  int vsize = 40000;
  int num_item = bsize * vsize;
  int stride = vsize;
  int num_seg = bsize;

  int warm = 10;
  int repeat = 100000;

  T * h_in = new T[num_item];
  init(h_in, num_item, stride);

  int * h_offset = new int[num_seg+1];
  for(int i = 0;i < num_seg + 1;i ++) {
    h_offset[i] = i * vsize;
  }

  // Allocate problem device arrays
  T *d_in = NULL;

  CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * num_item));

  // Init device input
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_item, cudaMemcpyHostToDevice));

  // Alloc device output array
  KVT *d_out = NULL;
  CubDebugExit(cudaMalloc((void**)&d_out, sizeof(KVT) * bsize));

  // Alloc & Init device offset array
  int * d_offset = NULL;
  CubDebugExit(cudaMalloc((void**)&d_offset, sizeof(int) * (num_seg + 1)));
  CubDebugExit(cudaMemcpy(d_offset, h_offset, sizeof(int) * (num_seg + 1), cudaMemcpyHostToDevice));

  // Request and allocate temporary storage
  void *d_tmp = NULL;
  size_t tmp_bytes = 0;
  CubDebugExit(DeviceSegmentedReduce::ArgMax(d_tmp, tmp_bytes, d_in, d_out, num_seg, d_offset, d_offset + 1));
  CubDebugExit(cudaMalloc((void**)&d_tmp, tmp_bytes));

  // RUN
  for(int i = 0;i < warm;i ++) {
    CubDebugExit(DeviceSegmentedReduce::ArgMax(d_tmp, tmp_bytes, d_in, d_out, num_seg, d_offset, d_offset + 1));
    CubDebugExit(cudaDeviceSynchronize());
  }

  std::cout << " WARM OK " << std::endl;

  uint64_t t0, t1;
  t0 = get_time();
  for(int i = 0;i < repeat;i ++) {
    CubDebugExit(DeviceSegmentedReduce::ArgMax(d_tmp, tmp_bytes, d_in, d_out, num_seg, d_offset, d_offset + 1));
    CubDebugExit(cudaDeviceSynchronize());
  }
  t1 = get_time();
  std::cout << (t1-t0) << " " << (t1 - t0)/repeat << " us." << std::endl;



  KVT* d_ret = get_out(d_out, num_seg);

  for(int i = 0;i < num_seg;i ++) {
    std::cout << d_ret[i].key << " " << d_ret[i].value << std::endl;
  }

  CubDebugExit(cudaDeviceSynchronize());

  return 0;
}
