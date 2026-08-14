
/*
 C Brent version from Numerical Recipes in C second edition
 */
#include <math.h>

#define FMAX(a,b) ({ \
    double _a = (a); \
    double _b = (b); \
    _a > _b ? _a : _b; \
})

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define SHFT(a, b, c, d) \
    (a) = (b);           \
    (b) = (c);           \
    (c) = (d);

double minimize_scalar(double start_a, double start_b, double (*func)(double, void*), void* params, double* xmin);
double brent(double ax, double bx, double cx, double (*func)(double, void*), void* context, double* xmin);
double minimize_scalar(double start_a, double start_b, double (*func)(double, void *), void *context, double *xmin) ;
