#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "functions.h"



__global__ void func_kernel(double * dy, double* a, double* base, double * params, int n)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
	uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x; 
//	printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
	// ensure we are within bounds
	double x[3] = {a[0] + base[0] * (0.5f + (double)idx), a[1] + base[1] * (0.5f + (double)idy), a[2] + base[2] * (0.5f + (double)idz),};
	__syncthreads();
	if (idx< n && idy<n && idz<n) {
		dy[offset] = F3(x, params) ;
		for (int j=0; j<3; j++) {
			dy[offset] *= base[j];
		}
	}
}
void cudasafe( cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

double Integrate(
    int functionCode, // Identifies the function (and dimensionality k)
    const double *a, // An array of k lower bounds
    const double *b, // An array of k upper bounds
    double n, // A target accuracy
    const double *params, // Parameters to function
    double *errorEstimate // Estimated error in integral
) 
{
	// size, in bytes, of each vector
	size_t bytes = (n*n*n)*sizeof(double);
		
	double *y = (double*)malloc(bytes);

//	double base[2] = {(b[0] - a[0]) / (double)n, (b[1] - a[1]) / (double)n};
	
	double base[3] = {(b[0] - a[0]) / (double)n, (b[1] - a[1]) / (double)n, (b[2] - a[2]) / (double)n};

	// allocate memory for each vector on GPU
	double * dy;
	double * dbase;
	double * da;
	double * dparams;
	
	cudaMalloc(&dy, bytes);
	cudaMalloc(&dbase, sizeof(base));
	cudaMalloc(&da, sizeof(a));
	cudaMalloc(&dparams, sizeof(params));

	// number of threads in each thread block
	int blockSize = 8;
	dim3 dimBlock(blockSize, blockSize, blockSize);

	// number of thread blocks in grid
	int gridSize = (int) ceil((double)n/blockSize);
	dim3 dimGrid(gridSize, gridSize, gridSize);
	
	cudaMemcpy(dbase, base, sizeof(base), cudaMemcpyHostToDevice);
	cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(dparams, params, sizeof(params), cudaMemcpyHostToDevice);

	//kernel execute
	func_kernel<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);
	
	//copy array back
	cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost);
	
	double sum = 0;
	for(uint32_t i=0; i<n*n*n; i++) {
		sum += y[i];
	}
	printf("final result: %0.10f\n", sum);

	cudaFree(dy);
	cudaFree(da);
	cudaFree(dbase);
	cudaFree(dparams);

	free(y);

	return sum;
}
void test1(void) {
	double a[2]={0,0};
	double b[2]={1,1};
	double params[2]={0.5,0.5};
	double error;
	int n = 32; 
	Integrate(1, a, b, n, params, &error); 	
}

void test2(void) {
	double exact=9.48557252267795;	// Correct to about 6 digits
	double a[3]={-1,-1,-1};
	double b[3]={1,1,1};
	int n = 8;	
	double error;
	Integrate(2, a, b, n, NULL, &error); 	
}

void test3(void) {
	double exact=-7.18387139942142f;	// Correct to about 6 digits
	double a[3]={0,0,0};
	double b[3]={5,5,5};
	double params[1]={2};
	int n = 512;	
	double error;
	Integrate(3, a, b, n, params, &error); 	
}

int main( int argc, char* argv[]) {
	test3();
}


