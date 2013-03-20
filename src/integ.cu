#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
//CUDA kernel

//__device__ float F1(float x){
//	return sin(x); 
//}


__global__ void func_kernel(func* f, float * dy, float a, float base, int n, int func_type)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = idx + idy * blockDim.x * gridDim.x;

	float params[2]={0.5,0.5};

	// ensure we are within bounds
	float x[1] = {a + base * ((float)0.5 + (float)idx, a + base * ((float)0.5 + (float)idx)};
	if (idx<n && idy<n)
		dy[idx*idy + idx] = F1(x, params ) * base;
	__syncthreads();
}
void cudasafe( cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

int main( int argc, char* argv[])
{
	int n = 4096; 
	// device input/output vectors
	
	// size, in bytes, of each vector
	size_t bytes = n*sizeof(float);
		
	float *y = (float*)malloc(bytes);
	
	float a, b;
	int functionCode = atoi(argv[1]);
	sscanf(argv[2], "%f", &a); 
	sscanf(argv[3], "%f", &b);
 
	float base = (b - a) / (float)n;
	printf("%f\n", base);
	printf("%f\n", a);

	// allocate memory for each vector on GPU
	float * dy;
	cudaMalloc(&dy, bytes);
	// allocate memory for params	
	// number of threads in each thread block
	int blockSize = 1024;
	dim3 dimBlock(blockSize, blockSize);

	// number of thread blocks in grid
	int gridSize = (int) ceil((float)n/blockSize);
	dim3 dimGrid(gridSize) = (gridSize, gridSize);
	
	//kernel execute
	func_kernel<<<dimGrid, dimBlock>>>(&dF0, dy, a, base, n, 0);
	
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

