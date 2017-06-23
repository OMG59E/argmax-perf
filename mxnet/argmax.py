import mxnet as mx
import numpy as np
import time


def main():
    bsize = 32
    vsize = 40000
    warm = 10
    repeat = 10
    tensor = mx.ndarray.ones((bsize, vsize), mx.gpu(0), dtype='float32')
    #tensor = mx.ndarray.ones((bsize, vsize), mx.cpu(0), dtype='float32')

    for i in range(warm):
        ret = mx.ndarray.argmax(tensor, axis=1)
        ret.wait_to_read()

    t0 = time.time()
    for i in range(repeat) :
        ret = mx.ndarray.argmax(tensor, axis=1)
        ret.wait_to_read()
    t1 = time.time()

    print (t1 - t0)
    print ((t1 - t0) / repeat) * 1e6

    print ret.shape
    print ret.asnumpy()

if __name__ == '__main__':
    main()
