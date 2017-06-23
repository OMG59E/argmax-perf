Compare argmax op in different system/libries
* Eigen : tensorflow's choice, with tensor support
* mxnet : handcrafted argmax, with tensor support
* cublas : amax, absolute max, not argmax, only support vector
* cub : nvidia-cub, support vector and batch-vector

cub has the best performance than others
