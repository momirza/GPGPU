#ifndef cw2_functions_h
#define cw2_functions_h

#include <math.h>

/* Description = exp(x)
	Code = 0     (this is the code that will be passed to Integrate to identify this function)
	k=1 (uni-variate)
	params = []  (no parameters)
*/
double F0(const double *x, const double *params)
{ return exp(x[0]); }

double myfunc(const double *x, const double *params)
{ return x[0] + x[1]; }

/* Name = sin(o1+x*y)*exp(o2+x)
	Code = 1
	k=2 (bi-variate)
	params = [o1,o2]  (2 parameters)
*/
double F1(const double *x, const double *params)
{
	return sin(params[0] + x[0]*x[1]) * exp(params[1]+x[0]);
}

/* Name = (round(exp(-x))-round(exp(y))*sin(z)
	Code = 2
	k=3 (tri-variate)
	params = []  (no parameters)
*/
double F2(const double *x, const double *params)
{
	return round(exp(-x[0]))-round(exp(x[1]))*sin(x[2]);
}

/* Name = Three-dimensional sinc
	Code = 3
	k=3 (tri-variate)
	params = [norm_lev]  (1 parameter)
*/
double F3(const double *x, const double *params)
{
	// Perform a vector norm of a specific power, e.g. if params[0]==2 then it is the euclidian norm
	float v=pow(x[0],params[0]) + pow(x[1],params[0]) + pow(x[2],params[0]);
	v=pow(v,1/params[0]);
	// Then a sinc
	return sin(v)/v;
}

/* Name = Gaussian CDF over 3 dimensions
	Code = 4
	k=3 (tri-variate)
	params= [sigma_0..sigma_8, scale]  (10 parameters)

	Sigma is the covariance matrix, but here we need the
	inverse of sigma. It's not worth calculating over and
	over, so we pre-calculate the inverse and store it in
	params. Don't worry too much about the details of the
	matrix, it isn't relative to making it go fast.
*/
double F4(const double *x, const double *params)
{
	// We need to calculate scale*exp(-x'*\Sigma^{-1}*x/2),
	// where \Sigma^{-1} is a 3x3 matrix.
	int i;
	float acc=0;
	for(i=0;i<3;i++){
		// We first form a row of the \Sigma^-1 * x
		float row= params[i*3+0]*x[0] + params[i*3+1]*x[1] + params[i*3+2]*x[2];
		// Then perform the left mutiply with x'
		acc += row*x[i];
	}
	return params[9] * exp(-acc / 2);
}

/* Name = Powerful Pony
	Code = 5
	k=3 (tri-variate)
	params= [] (no parameters)
*/
double F5(const double *x, const double *params)
{
	return powf(powf(sinf( pow(x[0], x[1]) ), 2), x[2]);
}

/* Name = Sphere Shell
	Code = 6
	k=3 (tri-variate)
	params= [radius,width]
*/
double F6(const double *x, const double *params)
{
	float a=powf(powf(x[0],2)+powf(x[1],2)+powf(x[2],2),0.5); // distance from origin
	float d=(a-params[0]);	// How far from surface of sphere
	if(d<0)
		d=-d;
	if(d<params[1]){
		return 1;
	}else{
		return 0;
	}
}


#endif
