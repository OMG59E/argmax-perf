CC ?= g++
NVCC ?= nvcc
CCFLAGS = -std=c++11 -O3
LDFLAGS = -L/usr/local/cuda/lib64  -lcudart -lstdc++ -lcublas -lglog
INCLUDES = 

all : a.out

a.out : argmax.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

argmax.o : argmax.cu
	$(NVCC) $(CCFLAGS) $(INCLUDES) -c  $< -o $@

.PHONY:
clean :
	rm -f *.o
	rm -f *.out
