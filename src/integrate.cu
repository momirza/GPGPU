#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "functions.h"
#include <time.h>

__global__ void func_kernel1d(float * dy, float* a, float* base, float * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t offset = idx; 
//      printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
        // ensure we are within bounds

        float x[3] = {a[0] + base[0] * (0.5f + idx)};
        if (idx< n) {
                dy[offset] = F0(x, params);
        }
}


__global__ void func_kernel2d(float * dy, float* a, float* base, float * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t offset = idx + idy * blockDim.x * gridDim.x;
//	printf("%d, %d, %d\n", idx, idy, offset);
        // ensure we are within bounds

        float x[2] = {a[0] + base[0] * (0.5f + idx), a[1] + base[1] * (0.5f + idy)};
        if (idx< n && idy<n) {
                dy[offset] = F1(x, params);
        }
}

__global__ void func_kernel3dF2(float * dy, float* a, float* base, float * params, int n)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
	uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x; 
//	printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
	// ensure we are within bounds

	float x[3] = {a[0] + base[0] * (0.5f + idx), a[1] + base[1] * (0.5f + idy), a[2] + base[2] * (0.5f + idz)};
	if (idx< n) {
		dy[offset] = F2(x, params) ;
	}
}

__global__ void func_kernel3dF3(float * dy, float* a, float* base, float * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
        uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x;
//      printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
        // ensure we are within bounds

        float x[3] = {a[0] + base[0] * (0.5f + idx), a[1] + base[1] * (0.5f + idy), a[2] + base[2] * (0.5f + idz)};
        if (idx< n) {
                dy[offset] = F3(x, params) ;
        }
}

__global__ void func_kernel3dF4(float * dy, float* a, float* base, float * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
        uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x;
//      printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
        // ensure we are within bounds

        float x[3] = {a[0] + base[0] * (0.5f + idx), a[1] + base[1] * (0.5f + idy), a[2] + base[2] * (0.5f + idz)};
        if (idx< n) {
                dy[offset] = F4(x, params) ;
        }
}

__global__ void func_kernel3dF5(float * dy, float* a, float* base, float * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
        uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x;
//      printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
        // ensure we are within bounds

        float x[3] = {a[0] + base[0] * (0.5f + idx), a[1] + base[1] * (0.5f + idy), a[2] + base[2] * (0.5f + idz)};
        if (idx< n) {
                dy[offset] = F5(x, params) ;
        }
}

__global__ void func_kernel3dF6(float * dy, float* a, float* base, float * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
        uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x;
//      printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
//	printf("%d, %d, %d, %d, %d\n", idx, idy, idz, offset, n);
        // ensure we are within bounds

        float x[3] = {a[0] + base[0] * (0.5f + idx), a[1] + base[1] * (0.5f + idy), a[2] + base[2] * (0.5f + idz)};
        if (idx< n) {
                dy[offset] = F6(x, params) ;
        }
}

__global__ void func_kernel3dF9(float * dy, float* a, float* base, float * params, int n)
{
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t idz = blockIdx.z * blockDim.z + threadIdx.z;
        uint32_t offset = idx + (idy + (blockDim.x * idz * gridDim.x))  * blockDim.x * gridDim.x;
//      printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
//        printf("%d, %d, %d, %d, %d\n", idx, idy, idz, offset, n);
        // ensure we are within bounds

        float x[3] = {a[0] + base[0] * (0.5f + idx), a[1] + base[1] * (0.5f + idy), a[2] + base[2] * (0.5f + idz)};
//	printf("%0.10f, %0.10f, %0.10f\n", x[0], x[1],x[2]);
//	printf("%0.10f, %0.10f, %0.10f\n", a[0], a[1],a[2]);
        if (idx< n) {
                dy[offset] = myfunc(x, params) ;
        }
}
void cudasafe( cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

double Integrate(
    int functionCode, // Identifies the function (and dimensionality k)
    const float *a, // An array of k lower bounds
    const float *b, // An array of k upper bounds
    float eps, // A target accuracy
    const float *params, // Parameters to function
    float *errorEstimate // Estimated error in integral
) 
{
    int i = 1; // multiplier
    *errorEstimate = 100; // set error to high value
    double sum = 0;
    double sum_temp = 0;
    while (*errorEstimate > eps) {
    	size_t freeMem = 0;
    	size_t totalMem = 0;
    	cudaMemGetInfo(&freeMem, &totalMem);  
    //	printf("Memory avaliable: Free: %lu, Total: %lu\n",freeMem, totalMem);
    	const int nsize = 10000000;
    	const int sz = sizeof(float) * nsize;
    	float *devicemem;
    	cudaMalloc((void **)&devicemem, sz);

    	cudaMemset(devicemem, 0, sz); // zeros all the bytes in devicemem
        int n;
        int k; int p = 0 ;  
        switch(functionCode){
            case 0: k=1;    p=0;    n=32*i; break;
            case 1: k=2;    p=2;    n=32*i;   break;
            case 2: k=3;    p=0;    n=8*i;   break;
            case 3: k=3;    p=1;    n=8*i;   break;
            case 4: k=3;    p=10;    n=8*i;   break;
            case 5: k=3;    p=0;    n=8*i;   break;
            case 6: k=3;    p=2;    n=128*i;   break;
            case 9: k=3;    p=0;    n=8*i;   break;
            default:
                fprintf(stderr, "Invalid function code.");
                exit(1);
        }
        
    	int n0=n, n1=n, n2=n;	// By default use n points in each dimension
    	// Collapse any dimensions we don't use
    	if(k<3){
    		n2=1;
    	}
    	if(k<2){
    		n1=1;
    	}
    	// size, in bytes, of each vector
        size_t bytes = (n0*n1*n2)*sizeof(float);
    	size_t bytes_temp = (pow(2,k)*n0*n1*n2)*sizeof(float);
    		
        float *y = (float*)malloc(bytes);
        float *y_temp = (float*)malloc(bytes_temp);
    	
        float base[3] = {(b[0] - a[0])/n, (b[1] - a[1])/n, (b[2] - a[2])/n};
    	float base_temp[3] = {(b[0] - a[0])/(n*2), (b[1] - a[1])/(n*2), (b[2] - a[2])/(n*2)};
    //	printf("base: %0.10f, %0.10f, %0.10f\n", base[0], base[1], base[2]);
    	// allocate memory for each vector on GPU
        float * dy;
        float * dy_temp;
        float * dbase;
    	float * dbase_temp;
    	float * da;
    	float * dparams;
    //	int  * dn;
    	
        cudaMalloc(&dy, bytes);
        cudaMalloc(&dy_temp, bytes_temp);
        cudaMalloc(&dbase, 3*sizeof(float));
    	cudaMalloc(&dbase_temp, 3*sizeof(float));
    //	cudaMalloc((void**)&dn, sizeof(int));	
    	cudaMalloc(&da, k*sizeof(int));
    	cudaMalloc(&dparams, p*sizeof(float));

        cudaMemcpy(dbase, base, 3*sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(dbase_temp, base_temp, 3*sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(da, a, k*sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(dparams, params, p*sizeof(float), cudaMemcpyHostToDevice);
    //	cudaMemcpy(dn,&n,sizeof(int), cudaMemcpyHostToDevice);

    	//kernel execute
    	if (k==1) {

    //		printf("1D\n");
    		// number of threads in each thread block
    		int blockSize = 32;
    		dim3 dimBlock(blockSize);

    		// number of thread blocks in grid
            int gridSize = (int) ceil((float)n/blockSize);
            dim3 dimGrid(gridSize);

            func_kernel1d<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);
            
    		int gridSize_temp = (int) ceil((float)n*2.0/blockSize);
    		dim3 dimGrid_temp(gridSize_temp);
    		func_kernel1d<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n);
    	}
    	else if (k==2) {
                    // number of threads in each thread block
    //		printf("2D\n");
                    int blockSize = 32;
                    dim3 dimBlock(blockSize, blockSize);

                    // number of thread blocks in grid
                    int gridSize = (int) ceil((float)n/blockSize);
                    dim3 dimGrid(gridSize, gridSize);

                    func_kernel2d<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);    
                    int gridSize_temp = (int) ceil((float)n*2.0/blockSize);
                    dim3 dimGrid_temp(gridSize_temp, gridSize_temp);
                    func_kernel2d<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n);	

    	}
    	else { 
                    // number of threads in each thread block
    //		printf("3D\n");
                    int blockSize = 8;
                    dim3 dimBlock(blockSize, blockSize, blockSize);

                    // number of thread blocks in grid
                    int gridSize = (int) ceil((float)n/blockSize);
                    dim3 dimGrid(gridSize, gridSize, gridSize);
                    int gridSize_temp = (int) ceil((float)n*2.0/blockSize);
                    dim3 dimGrid_temp(gridSize_temp, gridSize_temp, gridSize_temp);
                    if (functionCode==2) {
                        func_kernel3dF2<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);
                        func_kernel3dF2<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n);
                    }
                    else if (functionCode==3) {
                        func_kernel3dF3<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);
                        func_kernel3dF3<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n);
                    }
                    else if (functionCode==4) {
                        func_kernel3dF4<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);
                        func_kernel3dF4<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n);
                    }
                    else if (functionCode==5) {
                        func_kernel3dF5<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n);
                        func_kernel3dF5<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n);
                    }
                    else if (functionCode==6) {
                        func_kernel3dF6<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n); 
                        func_kernel3dF6<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n); 
                    }
                    else if (functionCode==9) {
                        func_kernel3dF9<<<dimGrid, dimBlock>>>(dy, da, dbase, dparams, n); 
                        func_kernel3dF9<<<dimGrid_temp, dimBlock>>>(dy_temp, da, dbase_temp, dparams, 2*n); 
                    }
                    else {
                        fprintf(stderr, "Invalid function code.");
    		}
    	}

    	
    	//copy array back
        cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(y_temp, dy_temp, bytes_temp, cudaMemcpyDeviceToHost);
    	
        sum = 0;
    	sum_temp = 0;
    	for(uint32_t i=0; i<n0*n1*n2; i++) {
            sum += y[i];
    	}
        for(uint32_t i=0; i<pow(2,k)*n0*n1*n2; i++) {
            sum_temp += y_temp[i];
        }
    	for(int j=0; j<k; j++) {
            sum *= base[j];
    		sum_temp *= base_temp[j];
        }
     //    printf("len: %0.10f\n", pow(2,k)*n0*n1*n2);
     //    printf("sum: %0.10f\n", sum);
    	// printf("sum_temp: %0.10f\n", sum_temp);


        cudaFree(dy);
    	cudaFree(dy_temp);
    	cudaFree(da);
        cudaFree(dbase);
    	cudaFree(dbase_temp);
    	cudaFree(dparams);

    //	cudaFree(dn);
        free(y);
        free(y_temp);
    	cudaMemset(devicemem, 0, sz); // zeros all the bytes in devicemem
        *errorEstimate = fabs(sum - sum_temp);
        i += 1;
    }
	return sum;
}

void testmyfunc(void) {
//        float a[3]={0,0,0};
        float a[3]={-1,-1,-1};
        float b[3]={2,2,2};
        float error;
        for (int n = 32; n<=1024; n*=2) {
		double time_spent;
		clock_t begin, end;
		begin = clock();
		Integrate(9, a, b, n, NULL, &error);
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//	printf("0 %d %0.10f", n, time_spent);	
	}
}

void test0(void) {
        float a[1]={0};
        float b[1]={1};
        float  error;
        // for (int n = 32; n<=1024; n*=2) {
		double time_spent;
		clock_t begin, end;
		begin = clock();
        float eps = 0.01;
		double result = Integrate(0, a, b, eps, NULL, &error);
        end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("********\n");
        printf("Error: %0.10f\n", error);
        printf("Result: %0.10f\n", result);
        printf("0 %0.10f\n", time_spent);

}

void test1(void) {
    float a[2]={0,0};
    float b[2]={1,1};
    float params[2]={0.5,0.5};
    float error;
    float eps = 2; 

    double time_spent;
    clock_t begin, end;
    begin = clock();
    double result = Integrate(1, a, b, eps, params, &error);    
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("********\n");
    printf("1 %0.10f\n", time_spent);
    printf("Error: %0.10f\n", error);
    printf("Result: %0.10f\n", result);

}

void test2(void) {
    float exact=9.48557252267795;   // Correct to about 6 digits
    float a[3]={-1,-1,-1};
    float b[3]={1,1,1};
    float eps = 0.01;   
    float error;
    double time_spent;
    clock_t begin, end;
    begin = clock();
    double result = Integrate(2, a, b, eps, NULL, &error);  
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("********\n");
    printf("2 %0.10f\n", time_spent);
    printf("Error: %0.10f\n", error);
    printf("Result: %0.10f\n", result);
}

void test3(void) {
    float exact=-7.18387139942142f; // Correct to about 6 digits
    float a[3]={0,0,0};
    float b[3]={5,5,5};
    float params[1]={2};
    float eps = 0.01;
    float error;
    double time_spent;
    clock_t begin, end;
    begin = clock();
    double result = Integrate(3, a, b, eps, params, &error);    
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("********\n");
    printf("3 %0.10f\n", time_spent);
    printf("Error: %0.10f\n", error);
    printf("Result: %0.10f\n", result);

}

void test4(void) {
        float exact=0.677779532970409f; // Correct to about 8 digits
    float a[3]={-16,-16,-16};   // We're going to cheat, and assume -16=-infinity.
    float b[3]={1,1,1};
    // We're going to use the covariance matrix with ones on the diagonal, and
    // 0.5 off the diagonal.
    const float PI=3.1415926535897932384626433832795f;
    float params[10]={
        1.5, -0.5, -0.5,
        -0.5, 1.5, -0.5,
        -0.5, -0.5, 1.5,
        pow(2*PI,-3.0/2.0)*pow(0.5,-0.5) // This is the scale factor
    };
    float eps = 0.01;
    float error;
    double time_spent;
    clock_t begin, end;
    begin = clock();
    double result = Integrate(4, a, b, eps, params, &error);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("********\n");
    printf("4 %0.10f\n", time_spent);
    printf("Error: %0.10f\n", error);
    printf("Result: %0.10f\n", result);

}
void test5(void) {
    float exact=13.4249394627056;   // Correct to about 6 digits
    float a[3]={0,0,0};
    float b[3]={3,3,3};
    float eps = 0.01;
    float error;
    double time_spent;
    clock_t begin, end;
    begin = clock();
    double result = Integrate(5, a, b, eps, NULL, &error);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("********\n");
    printf("5 %0.10f\n", time_spent);
    printf("Error: %0.10f\n", error);
    printf("Result: %0.10f\n", result);
}

void test6(void) {

    float exact=   2.261955088165;
    float a[3]={-4,-4,-4};
    float b[3]={4,4,4};
    float params[2]={3,0.01};
    float eps = 0.1;
    float error;
    double time_spent;
    clock_t begin, end;
    begin = clock();
    double result = Integrate(6, a, b, eps, params, &error);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("********\n");
    printf("6 %0.10f\n", time_spent);
    printf("Error: %0.10f\n", error);
    printf("Result: %0.10f\n", result);
}

int main( int argc, char* argv[]) {
    // testmyfunc();
    test0(); // works
    test1(); // works
    test2(); // works
    test3(); // works
    test4(); // works
    test5(); // works
    test6(); // works
    return 0;
}


