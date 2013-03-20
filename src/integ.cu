#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
//CUDA kernel

//__device__ float F1(float x){
//	return sin(x); 
//}

__global__ void func_kernel(float * dy, float a, float base, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// ensure we are within bounds
	float x[1] = {a + base * ((float)0.5 + (float)id)};
	if (id<n)
		dy[id] = base * F0(x, NULL);
	__syncthreads();
}


int main( int argc, char* argv[])
{
	// define a function pointer type
	//typedef float (*func)(int);
	//__device__ func dF1 = F1;
	
	// vector size
	int n = 4096; 
	// device input/output vectors
	
	// size, in bytes, of each vector
	size_t bytes = n*sizeof(float);
		
	float *y = (float*)malloc(bytes);
	
	float a, b;
	sscanf(argv[1], "%f", &a); 
	sscanf(argv[2], "%f", &b);
 
	float base = (b - a) / (float)n;
	printf("%f\n", base);
	printf("%f\n", a);

	// allocate memory for each vector on GPU
	float * dy;
	cudaMalloc(&dy, bytes);
	
	// number of threads in each thread block
	int blockSize = 1024;

	// number of thread blocks in grid
	int gridSize = (int) ceil((float)n/blockSize);
	
	//kernel execute
	func_kernel<<<gridSize, blockSize>>>(dy, a, base, n);
	
	//copy array back
	cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost);
	
	float sum = 0;

	for(int i=0; i<n; i++) {
		sum += y[i];
	}
	printf("final result: %f\n", sum);

	cudaFree(dy);
	
	free(y);
	return 0;
}

