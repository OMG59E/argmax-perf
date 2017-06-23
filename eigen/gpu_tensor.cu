
#define EIGEN_USE_GPU

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
#include "./util.h"

#define layout (Eigen::RowMajor)

using Eigen::Tensor;

int main(int argc, char ** argv) {

  int bsize = 32;
  int vsize = 40000;
  Eigen::array<Eigen::DenseIndex, 2> shape = {bsize, vsize};
  //int layout = Eigen::RowMajor;
  int warm = 10;
  int repeat = 10;
  int dim = 1;

  Tensor<double, 2, layout> in (bsize, vsize);
  Tensor<Eigen::DenseIndex, 1, layout> out_max(bsize);

  in.setRandom();
  in *= in.constant(100.0);
  in(0, 0) = -1000.0;
  in(bsize-1, vsize-1) = 1000.0;
  
  size_t in_bytes = in.size() * sizeof(double);
  size_t out_bytes = out_max.size() * sizeof(Eigen::DenseIndex);

  std::cout << " init ok " << std::endl;
  double * d_in;
  Eigen::DenseIndex * d_out_max;
  cudaMalloc((void**)(&d_in), in_bytes);
  cudaMalloc((void**)(&d_out_max), out_bytes);

  cudaMemcpy(d_in, in.data(), in_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<double, 2, layout>, Eigen::Aligned > gpu_in(d_in, bsize, vsize);
  //Eigen::TensorMap<Eigen::Tensor<double, 2, layout> > gpu_in(d_in, shape);
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1, layout> > gpu_out_max(d_out_max, 1);


  for(int i = 0;i < warm;i ++) {
    gpu_out_max.device(gpu_device) = gpu_in.argmax(dim);
    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
  }

  uint64_t t0, t1;
  t0 = get_time();
  for(int i = 0;i < repeat;i ++) {
    gpu_out_max.device(gpu_device) = gpu_in.argmax(dim);
  }
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
  t1 = get_time();

  std::cout <<  (t1 - t0) << " " << (t1 - t0) / repeat << " us " << std::endl;

  assert(cudaMemcpyAsync(out_max.data(), d_out_max, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  std::cout << out_max << std::endl;


  return 0;
}




