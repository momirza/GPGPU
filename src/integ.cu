#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "functions.h"

__global__ void func_kernel1d(double * dy, double* a, double* base, double * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t offset = idx; 
//      printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
        // ensure we are within bounds

        double x[3] = {a[0] + base[0] * (0.5f + (double)idx)};
        if (idx< n) {
                dy[offset] = F0(x, params);
//                for (int j=0; j<1; j++) {
//                        dy[offset] *= base[j];
//                }
        }
}


__global__ void func_kernel2d(double * dy, double* a, double* base, double * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t offset = idx + idy * blockDim.x * gridDim.x;
//	printf("%d, %d, %d\n", idx, idy, offset);
        // ensure we are within bounds

        double x[3] = {a[0] + base[0] * (0.5f + (double)idx), a[1] + base[1] * (0.5f + (double)idy)};
        if (idx< n && idy<n) {
                dy[offset] = F1(x, params);
//		printf("%0.5f\n", dy[offset]);
//                for (int j=0; j<2; j++) {
//			printf("Base: %0.5f\n", base[j]);
//                        dy[offset] *= base[j];
//                }
        }
}


__global__ void func_kernel3d(double * dy, double* a, double* base, double * params, int n)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
	uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x; 
//	printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
	// ensure we are within bounds

	double x[3] = {a[0] + base[0] * (0.5f + (double)idx), a[1] + base[1] * (0.5f + (double)idy), a[2] + base[2] * (0.5f + (double)idz)};
	if (idx< n) {
		dy[offset] = F5(x, params) ;
//		for (int j=0; j<3; j++) {
//			dy[offset] *= base[j];
//		}
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
	int n0=n, n1=n, n2=n;	// By default use n points in each dimension
	int k;	
	switch(functionCode){
		case 0:	k=1;	break;
		case 1:	k=2;	break;
		case 2:	k=3;	break;
		case 3:	k=3;	break;
		case 4:	k=3;	break;
		case 5:	k=3;	break;
		case 6:	k=3;	break;
		default:
			fprintf(stderr, "Invalid function code.");
			exit(1);
	}
	
	// Collapse any dimensions we don't use
	if(k<3){
		n2=1;
	}
	if(k<2){
		n1=1;
	}
	// size, in bytes, of each vector
	size_t bytes = (n0*n1*n2)*sizeof(double);
		
	double *y = (double*)malloc(bytes);
	
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

	cudaMemcpy(dbase, base, sizeof(base), cudaMemcpyHostToDevice);
	cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(dparams, params, sizeof(params), cudaMemcpyHostToDevice);


	//kernel execute
	if (k==1) {

		// number of threads in each thread block
		int blockSize = 32;
		dim3 dimBlock(blockSize);

		// number of thread blocks in grid
		int gridSize = (int) ceil((double)n/blockSize);
		dim3 dimGrid(gridSize);

		func_kernel1d<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);
	}
	else if (k==2) {
                // number of threads in each thread block
                int blockSize = 32;
                dim3 dimBlock(blockSize, blockSize);

                // number of thread blocks in grid
                int gridSize = (int) ceil((double)n/blockSize);
                dim3 dimGrid(gridSize, gridSize);

                func_kernel2d<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);	

	}
	else { 
                // number of threads in each thread block
                int blockSize = 8;
                dim3 dimBlock(blockSize, blockSize, blockSize);

                // number of thread blocks in grid
                int gridSize = (int) ceil((double)n/blockSize);
                dim3 dimGrid(gridSize, gridSize, gridSize);

                func_kernel3d<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);	
	}

	
	//copy array back
	cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost);
	
	double sum = 0;
	for(uint32_t i=0; i<n0*n1*n2; i++) {
		sum += y[i];
	}
	for(int j=0; j<k; j++)
		sum *= base[j];
	printf("final result: %0.10f\n", sum);

	cudaFree(dy);
	cudaFree(da);
	cudaFree(dbase);
	cudaFree(dparams);

	free(y);

	return sum;
}

void test0(void) {
        double a[1]={0};
        double b[1]={1};
        double error;
        int n = 32;
        Integrate(0, a, b, n, NULL, &error);
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
	int n = 256;	
	double error;
	Integrate(2, a, b, n, NULL, &error); 	
}

void test3(void) {
	double exact=-7.18387139942142f;	// Correct to about 6 digits
	double a[3]={0,0,0};
	double b[3]={5,5,5};
	double params[1]={2};
	int n = 256;	
	double error;
	Integrate(3, a, b, n, params, &error); 	
}

void test4(void) {
        double exact=0.677779532970409f;	// Correct to about 8 digits
	double a[3]={-16,-16,-16};	// We're going to cheat, and assume -16=-infinity.
	double b[3]={1,1,1};
	// We're going to use the covariance matrix with ones on the diagonal, and
	// 0.5 off the diagonal.
	const double PI=3.1415926535897932384626433832795f;
	double params[10]={
		1.5, -0.5, -0.5,
		-0.5, 1.5, -0.5,
		-0.5, -0.5, 1.5,
		pow(2*PI,-3.0/2.0)*pow(0.5,-0.5) // This is the scale factor
	};
	int n = 64;
	double error;
        Integrate(4, a, b, n, params, &error);
}
void test5(void) {
	double exact=13.4249394627056;	// Correct to about 6 digits
	double a[3]={0,0,0};
	double b[3]={3,3,3};
        int n = 256;
        double error;
        Integrate(5, a, b, n, NULL, &error);
}
void test6(void) {

	double exact=   2.261955088165;
	double a[3]={-4,-4,-4};
	double b[3]={4,4,4};
	double params[2]={3,0.01};
        int n = 256;
        double error;
        Integrate(6, a, b, n, params, &error);
}

int main( int argc, char* argv[]) {
	test0();
	test1();
}


