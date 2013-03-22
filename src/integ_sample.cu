#include "functions.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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
//  printf("%d, %d, %d\n", idx, idy, offset);
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
//  printf("%d, %d, %d, %d\n", idx, idy, idz, offset);
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
//  printf("%d, %d, %d, %d, %d\n", idx, idy, idz, offset, n);
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
//  printf("%0.10f, %0.10f, %0.10f\n", x[0], x[1],x[2]);
//  printf("%0.10f, %0.10f, %0.10f\n", a[0], a[1],a[2]);
        if (idx< n) {
                dy[offset] = myfunc(x, params) ;
        }
}


void cudasafe( cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

extern "C" double integrate_cu(
    int functionCode, // Identifies the function (and dimensionality k)
    const float *a, // An array of k lower bounds
    const float *b, // An array of k upper bounds
    float eps, // A target accuracy
    const float *params, // Parameters to function
    float *errorEstimate // Estimated error in integral
) 
{
    int mult = 1; // multiplier
    *errorEstimate = 100; // set error to high value
    double sum = 0;
    double sum_temp = 0;
    while (*errorEstimate > eps) {
        size_t freeMem = 0;
        size_t totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);  
    //  printf("Memory avaliable: Free: %lu, Total: %lu\n",freeMem, totalMem);
        const int nsize = 10000000;
        const int sz = sizeof(float) * nsize;
        float *devicemem;
        cudaMalloc((void **)&devicemem, sz);

        cudaMemset(devicemem, 0, sz); // zeros all the bytes in devicemem
        int n;
        int k; int p = 0 ;  
        switch(functionCode){
            case 0: k=1;    p=0;    n=32*mult; break;
            case 1: k=2;    p=2;    n=32*mult;   break;
            case 2: k=3;    p=0;    n=8*mult;   break;
            case 3: k=3;    p=1;    n=8*mult;   break;
            case 4: k=3;    p=10;    n=8*mult;   break;
            case 5: k=3;    p=0;    n=8*mult;   break;
            case 6: k=3;    p=2;    n=128*mult;   break;
            case 9: k=3;    p=0;    n=8*mult;   break;
            default:
                fprintf(stderr, "Invalid function code.");
                exit(1);
        }
        
        int n0=n, n1=n, n2=n;   // By default use n points in each dimension
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
    //  printf("base: %0.10f, %0.10f, %0.10f\n", base[0], base[1], base[2]);
        // allocate memory for each vector on GPU
        float * dy;
        float * dy_temp;
        float * dbase;
        float * dbase_temp;
        float * da;
        float * dparams;
    //  int  * dn;
        
        cudaMalloc(&dy, bytes);
        cudaMalloc(&dy_temp, bytes_temp);
        cudaMalloc(&dbase, 3*sizeof(float));
        cudaMalloc(&dbase_temp, 3*sizeof(float));
    //  cudaMalloc((void**)&dn, sizeof(int));   
        cudaMalloc(&da, k*sizeof(int));
        cudaMalloc(&dparams, p*sizeof(float));

        cudaMemcpy(dbase, base, 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dbase_temp, base_temp, 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(da, a, k*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dparams, params, p*sizeof(float), cudaMemcpyHostToDevice);
    //  cudaMemcpy(dn,&n,sizeof(int), cudaMemcpyHostToDevice);

        //kernel execute
        if (k==1) {

    //      printf("1D\n");
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
    //      printf("2D\n");
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
    //      printf("3D\n");
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

    //  cudaFree(dn);
        free(y);
        free(y_temp);
        cudaMemset(devicemem, 0, sz); // zeros all the bytes in devicemem
        *errorEstimate = fabs(sum - sum_temp);
        mult += 1;
    }
    return sum;
}
