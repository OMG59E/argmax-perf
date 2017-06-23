CC ?= nvcc
CCFLAGS = -std=c++11 -O2
LDFLAGS = -L/usr/local/cuda/lib64  -lcudart -lstdc++
INCLUDES = 

all : a.out

a.out : gpu_tensor.o
	g++ $(CCFLAGS) -o $@ $^ $(LDFLAGS)

gpu_tensor.o : gpu_tensor.cu
	nvcc $(CCFLAGS) $(INCLUDES) -c  $< -o $@

.PHONY:
clean :
	rm -f *.o
	rm -f *.out
