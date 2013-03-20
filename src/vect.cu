#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//CUDA kernel
__global__ void vecAdd(float *a, float *b, float *c, int  n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// ensure we are within bounds
	if (id<n)
		c[id] = a[id] + b[id];
}

int main( int argc, char* argv[])
{
	// vector size
	int n = 2000;

	// device input/output vectors
	float *d_a;
	float *d_b;
	float *d_c;
	
	// size, in bytes, of each vector
	size_t bytes = n*sizeof(float);
		
	float *h_a = (float*)malloc(bytes);
	float *h_b = (float*)malloc(bytes);
	float *h_c = (float*)malloc(bytes);
	// allocate memory for each vector on GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	int i;
	// initialize vectors
	for (i=0; i< n; i++) {
		h_a[i] = sin(i)*sin(i);
		h_b[i] = cos(i)*cos(i);
	}

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
	

	// number of threads in each thread block
	int blockSize = 1024;

	// number of thread blocks in grid
	int gridSize = (int) ceil((float)n/blockSize);
	
	//kernel execute
	vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
	
	//copy array back
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
	
	float sum = 0;

	for(i=0; i<n; i++) {
		sum += h_c[i];
		printf("%0.2f\n", h_c[i]);
	}
	printf("final result: %f\n", sum/(float)n);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	free(h_a);
	free(h_b);
	free(h_c);
	return 0;
}

