CC ?= g++
CCFLAGS = -std=c++11 -O2
LDFLAGS = -L/usr/local/cuda/lib64  -lcudart -lstdc++ -lcublas
INCLUDES = 

all : a.out

a.out : argmax.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

argmax.o : argmax.cc
	$(CC) $(CCFLAGS) $(INCLUDES) -c  $< -o $@

.PHONY:
clean :
	rm -f *.o
	rm -f *.out
