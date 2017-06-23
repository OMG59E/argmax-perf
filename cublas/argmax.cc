#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include "util.h"



void findMaxAndMinGPU(double* values, int* max_idx, int* min_idx, int n)
{
    double* d_values;
    cublasHandle_t handle;
    cublasStatus_t stat;
    cudaMalloc((void**) &d_values, sizeof(double) * n);
    cudaMemcpy(d_values, values, sizeof(double) * n, cudaMemcpyHostToDevice);
    cublasCreate(&handle);

    cudaError_t err;
    int warm = 10;
    int repeat = 10;
    for(int i = 0;i < warm ;i ++) {
      stat = cublasIdamax(handle, n, d_values, 1, max_idx);
      if (stat != CUBLAS_STATUS_SUCCESS)
          printf("Max failed\n");
    }
    if (cudaDeviceSynchronize()!= cudaSuccess) {
      printf("Sync Failed!\n");
    }


    std::cout << " BEGIN " << std::endl;
    uint64_t t0, t1;
    t0 = get_time();
    for(int i = 0;i < repeat ;i ++) {
      stat = cublasIdamax(handle, n, d_values, 1, max_idx);
      if (stat != CUBLAS_STATUS_SUCCESS)
          printf("Max failed\n");
      if (cudaDeviceSynchronize()!= cudaSuccess) {
        printf("Sync Failed!\n");
      }
    }
    t1 = get_time();
    std::cout << (t1 - t0) << " " << (t1 - t0) / repeat << std::endl;

    std::cout << " END " << std::endl;

    cudaFree(d_values);
    cublasDestroy(handle);
}

int main(void)
{
    const int vmax=1000, nvals=40000;

    double vals[nvals];
    srand ( time(NULL) );
    for(int j=0; j<nvals; j++) {
      vals[j] = double(-j);
       //vals[j] = float(rand() % vmax);
    }

    int minIdx, maxIdx;
    findMaxAndMinGPU(vals, &maxIdx, &minIdx, nvals);

    int cmin = 0, cmax=0;
    for(int i=1; i<nvals; i++) {
        cmin = (vals[i] < vals[cmin]) ? i : cmin;
        cmax = (vals[i] > vals[cmax]) ? i : cmax;
    }

    fprintf(stdout, "%d %d %d %d\n", minIdx, cmin, maxIdx, cmax);

    return 0;
}


