#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

extern double integrate_cu(
    int functionCode, // Identifies the function (and dimensionality k)
    const float *a, // An array of k lower bounds
    const float *b, // An array of k upper bounds
    float eps, // A target accuracy
    const float *params, // Parameters to function
    float *errorEstimate);

double Integrate(
    int functionCode, // Identifies the function (and dimensionality k)
    const float *a, // An array of k lower bounds
    const float *b, // An array of k upper bounds
    float eps, // A target accuracy
    const float *params, // Parameters to function
    float *errorEstimate
    ) {
        return integrate_cu(functionCode, a, b, eps, params, errorEstimate);
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


