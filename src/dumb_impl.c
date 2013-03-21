#include "functions.h"

#include <stdio.h>
#include <stdlib.h>

/*! This is a simple example of multi-dimensional integration
	using a simple (not necessarily optimal) spacing of points.
	Note that this doesn't perform any error estimation - it
	only calculates the value for a given grid size.
*/
double IntegrateExample(
  int functionCode,
  int n,	// How many points on each dimension
  const double *a, // An array of k lower bounds
  const double *b, // An array of k upper bounds
  const double *params // Parameters to function
){
	int k=-1, total=-1, i0, i1, i2, j;
	// Accumulate in double, as it avoids doubleing-point errors when adding large
	// numbers of values together. Note that using double in a GPU has implications,
	// as some GPUs cannot do doubles, and on others they are much slower than doubles
	double acc=0;	
	double *x=NULL;
	int n0=n, n1=n, n2=n;	// By default use n points in each dimension
	
	switch(functionCode){
		case 0:	k=1;	break;
		case 1:	k=2;	break;
		case 2:	k=3;	break;
		case 3:	k=3;	break;
		case 4:	k=3;	break;
		case 5:	k=3;	break;
		case 6:	k=3;	break;
		case 9:	k=3;	break;
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
	
	x=(double *)malloc(sizeof(double)*k);

	// Loop over highest dimension on outside, as it might be collapsed to zero
	for(i2=0;i2<n2;i2++){
		if(k>=2){
			x[2]=a[2]+(b[2]-a[2]) * (i2+0.5f)/n2;
		}
		
		for(i1=0;i1<n1;i1++){
			if(k>=1){
				x[1]=a[1]+(b[1]-a[1]) * (i1+0.5f)/n1;
			}
			
			// Inner dimension is never collapsed to zero
			for(i0=0;i0<n0;i0++){
				x[0]=a[0]+(b[0]-a[0]) * (i0+0.5f)/n0;
				
				// Now call the function. Note that it is rather
				// inefficient to be choosing the function in the inner loop...
				switch(functionCode){
				case 0:	acc+=F0(x,params);	break;
				case 1:	acc+=F1(x,params);	break;
				case 2:	acc+=F2(x,params);	break;
				case 3:	acc+=F3(x,params);	break;
				case 4:	acc+=F4(x,params);	break;
				case 5:	acc+=F5(x,params);	break;
				case 6:	acc+=F6(x,params);	break;
				case 9:	acc+=myfunc(x,params);	break;
				}
			}
		}
	}
	
	free(x);
	
	// Do the final normalisation and return the results
	for(j=0;j<k;j++){
		acc=acc*(b[j]-a[j]);
	}
	return acc/(n0*n1*n2);
}

void Test0()
{
	double exact=(exp(1)-1);	// Exact result
	double a[1]={0};
	double b[1]={1};
	int n;
	
	for(n=2;n<=512;n*=2){		
		double res=IntegrateExample(
		  0, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  NULL // Parameters to function (no parameters for this function)
		);
		fprintf(stderr, "F0, n=%d, value=%lf, error=%lg\n", n, res, res-exact);
	}
}

void mytest()
{
	double exact=16;	// Exact result
	double a[3]={-1, -1, -1};
	double b[3]={2, 2, 2};
	int n;
	
	for(n=2;n<=512;n*=2){		
		double res=IntegrateExample(
		  9, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  NULL // Parameters to function (no parameters for this function)
		);
		fprintf(stderr, "mytest, n=%d, value=%lf, error=%lg\n", n, res, res-exact);
	}
}

void Test1()
{
	double exact=1.95683793560212f;	// Correct to about 10 digits
	double a[2]={0,0};
	double b[2]={1,1};
	double params[2]={0.5,0.5};
	int n;
	
	for(n=2;n<=1024;n*=2){		
		double res=IntegrateExample(
		  1, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  params // Parameters to function
		);
		fprintf(stderr, "F1, n=%d, value=%lf, error=%lg\n", n, res, res-exact);
	}
}

void Test2()
{
	double exact=9.48557252267795;	// Correct to about 6 digits
	double a[3]={-1,-1,-1};
	double b[3]={1,1,1};
	int n;
	
	for(n=2;n<=256;n*=2){		
		double res=IntegrateExample(
		  2, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  NULL // Parameters to function (no parameters for this function)
		);
		fprintf(stderr, "F2, n=%d, value=%lf, error=%lg\n", n, res, res-exact);
	}
}

void Test3()
{
	double exact=-7.18387139942142f;	// Correct to about 6 digits
	double a[3]={0,0,0};
	double b[3]={5,5,5};
	double params[1]={2};
	int n;
	
	for(n=2;n<=256;n*=2){		
		double res=IntegrateExample(
		  3, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  params // Parameters to function (no parameters for this function)
		);
		fprintf(stderr, "F3, n=%d, value=%lf, error=%lg\n", n, res, res-exact);
	}
}

void Test4()
{
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
	int n;
	
	for(n=2;n<=512;n*=2){		
		double res=IntegrateExample(
		  4, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  params // Parameters to function (no parameters for this function)
		);
		fprintf(stderr, "F4, n=%d, value=%lf, error=%lg	\n", n, res, res-exact);
	}
}

void Test5()
{
	double exact=13.4249394627056;	// Correct to about 6 digits
	double a[3]={0,0,0};
	double b[3]={3,3,3};
	int n;
	
	for(n=2;n<=512;n*=2){		
		double res=IntegrateExample(
		  5, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  NULL
		);
		fprintf(stderr, "F5, n=%d, value=%lf, error=%lg	\n", n, res, res-exact);
	}
}

void Test6()
{
	// Integrate over a shell with radius 3 and width 0.02
	//  = volume of a sphere of 3.01 minus a sphere of 2.99
	double exact=   2.261955088165;
	double a[3]={-4,-4,-4};
	double b[3]={4,4,4};
	double params[2]={3,0.01};
	int n;
	
	for(n=2;n<=2048;n*=2){		
		double res=IntegrateExample(
		  6, // functionCode,
		  n,	// How many points on each dimension
		  a, // An array of k lower bounds
		  b, // An array of k upper bounds
		  params
		);
		fprintf(stderr, "F6, n=%d, value=%lf, error=%lg	\n", n, res, res-exact);
	}
}

int main(int argc, char *argv[])
{
	mytest();
	Test0();
	Test1();
	Test2();
	Test3();
	Test4();
	Test5();
	Test6();

	return 0;
}
