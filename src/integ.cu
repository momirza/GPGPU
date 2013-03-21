#include <stdio.h>
#include <stdlib.h>
#include "functions.h"



__global__ void func_kernel(float * dy, float* a, float* base, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = idx + idy * blockDim.x * gridDim.x; 
	float params[2]={0.5,0.5};

	// ensure we are within bounds
	float x[2] = {a[0] + base[0] * (0.5f + (float)idx), a[1] + base[1] * (0.5f + (float)idy)};
	__syncthreads();
	if (idx< n && idy<n) {
		dy[offset] = F1(x, params) ;
		for (int j=0; j<2; j++) {
			dy[offset] *= base[j];
		}
	}
}
void cudasafe( cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

int main( int argc, char* argv[])
{
	//int n = 32; 
	// device input/output vectors
	int n = atoi(argv[1]);
	
	// size, in bytes, of each vector
	size_t bytes = (n*n)*sizeof(float);
		
	float *y = (float*)malloc(bytes);
	
 	
	float a[2]={0,0};
	float b[2]={1,1};

	float base[2] = {(b[0] - a[0]) / (float)n, (b[1] - a[1]) / (float)n};

	// allocate memory for each vector on GPU
	float * dy;
	float * dbase;
	float * da;
	cudaMalloc(&dy, bytes);
	cudaMalloc(&dbase, sizeof(base));
	cudaMalloc(&da, sizeof(a));
	// allocate memory for params	
	// number of threads in each thread block
	int blockSize = 32;
	dim3 dimBlock(blockSize, blockSize);

	// number of thread blocks in grid
	int gridSize = (int) ceil((float)n/blockSize);
	dim3 dimGrid(gridSize, gridSize);
	
	cudaMemcpy(dbase, base, sizeof(base), cudaMemcpyHostToDevice);
	cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice);
	//kernel execute
	func_kernel<<<dimGrid, dimBlock>>>(dy, da, dbase, n);
	
	//copy array back
	cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost);
	
	float sum = 0;

	for(int i=0; i<n*n; i++) {
		sum += y[i];
	}
	printf("final result: %f\n", sum);

	cudaFree(dy);
	cudaFree(da);
	cudaFree(dbase);

	free(y);

	return 0;
}


