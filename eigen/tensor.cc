#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include "./util.h"

int main(int argc, char ** argv) {

  int bsize = 32;
  int vsize = 40000;
  
  int warm = 10;
  int repeat = 100;

  Eigen::Tensor<double, 2, Eigen::RowMajor>  tensor(bsize, vsize);
  tensor.setRandom();


  
  Eigen::Tensor<Eigen::DenseIndex, 1, Eigen::RowMajor> tensor_argmax;
  int dim = 1;

  for(int i = 0;i < warm;i ++) {
    tensor_argmax = tensor.argmax(dim);
  }

  uint64_t t0, t1;
  t0 = get_time();
  for(int i = 0;i < repeat; i ++) {
    tensor_argmax = tensor.argmax(dim);
  }
  t1 = get_time();

  std::cout << (t1 - t0) / repeat << " us " << std::endl;

  std::cout << tensor_argmax << std::endl;
  //std::cout << tensor_argmax << std::endl;

  return 0;
}
