all:
	nvcc -c -arch=sm_20 integ_sample.cu
	gcc --std=c99 -I/usr/local/cuda/include/ -o a.out -L/usr/local/cuda/lib64/ integrate.c integ_sample.o -lcuda -lm -lcudart
