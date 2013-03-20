#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
//CUDA kernel

//__device__ float F1(float x){
//	return sin(x); 
//}

// define a function pointer type
typedef float (*func)(const float *,const float *);
__device__ func dF0 = F0;
__device__ func dF1 = F1;
__device__ func dF2 = F2;
__device__ func dF3 = F3;
__device__ func dF4 = F4;
__device__ func dF5 = F5;
__device__ func dF6 = F6;

__global__ void func_kernel(func* F, float * dy, float a, float base, int n, int func_type)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// ensure we are within bounds
	float x[1] = {a + base * ((float)0.5 + (float)id)};
	if (id<n)
		dy[id] = base * F[func_type](x, NULL);
	__syncthreads();
}


int main( int argc, char* argv[])
{
	// allocatiing function pointers
	int func_count = 7;
	func * d_f;
	func * h_f;	
	h_f = (func*)malloc(func_count*sizeof(func));
	cudaMalloc((void**)&d_f, func_count*sizeof(func));
	cudaMemcpyFromSymbol( &h_f[0], dF0, sizeof(func));	
	cudaMemcpyFromSymbol( &h_f[1], dF1, sizeof(func));	
	cudaMemcpyFromSymbol( &h_f[2], dF2, sizeof(func));	
	cudaMemcpyFromSymbol( &h_f[3], dF3, sizeof(func));	
	cudaMemcpyFromSymbol( &h_f[4], dF4, sizeof(func));	
	cudaMemcpyFromSymbol( &h_f[5], dF5, sizeof(func));	
	cudaMemcpyFromSymbol( &h_f[6], dF6, sizeof(func));	
	//	   dst				
	//	    |	src
	//	    |	 |	size
	cudaMemcpy(d_f, h_f, func_count*sizeof(func), cudaMemcpyHostToDevice);

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
	func_kernel<<<gridSize, blockSize>>>(d_f, dy, a, base, n, 0);
	
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

