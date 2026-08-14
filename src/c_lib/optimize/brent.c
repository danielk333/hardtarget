#include "brent.h"

#include <math.h>

// brent
#define ITMAX 100                    // Maximum iterations
#define CGOLD 0.3819660112501051518  // Golden ratio
#define ZEPS 1.0e-11
#define XTOL 1.4901161193847656e-8  // Tolerance, square root of machine epsilon for double-precision floats

// mnbrack
#define GOLD 1.618034  // Default ratio by which successive intervals are magnified
#define GLIMIT 100.0   // Maximum magnification allowed for a parabolic-fit step
#define TINY 1.0e-2

double brent(double ax, double bx, double cx, double (*f)(double, void*), void* params, double* xmin) {
    /*
     Given a function f, and given a bracketing triplet of abscissas ax, bx, cx (such that bx is
     between ax and cx, and f(bx) is less than both f(ax) and f(cx)), this routine isolates
     the minimum to a fractional precision of about xtol using Brent’s method. The abscissa of
     the minimum is returned as xmin, and the minimum function value is returned as brent, the
     returned function value. parameters is a struct of the function parameters not used for the
     optimization
    */
    int iter;
    double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
    double e = 0.0;

    a = (ax < cx ? ax : cx);
    b = (ax > cx ? ax : cx);
    x = w = v = bx;
    fw = fv = fx = (*f)(x, params);
    for (iter = 1; iter <= ITMAX; iter++) {
        xm = 0.5 * (a + b);
        tol2 = 2.0 * (tol1 = XTOL * fabs(x) + ZEPS);
        if (fabs(x - xm) <= (tol2 - 0.5 * (b - a))) {
            *xmin = x;
            return fx;
        }
        if (fabs(e) > tol1) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) {
                p = -p;
            }
            q = fabs(q);
            etemp = e;
            e = d;
            if (fabs(p) >= fabs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
                d = CGOLD * (e = (x >= xm ? a - x : b - x));
            } else {
                d = p / q;
                u = x + d;
                if (u - a < tol2 || b - u < tol2) {
                    d = SIGN(tol1, xm - x);
                }
            }
        } else {
            d = CGOLD * (e = (x >= xm ? a - x : b - x));
        }
        u = (fabs(d) >= tol1 ? x + d : x + SIGN(tol1, d));
        fu = (*f)(u, params);
        if (fu <= fx) {
            if (u >= x) {
                a = x;
            } else {
                b = x;
            }
            SHFT(v, w, x, u);
            SHFT(fv, fw, fx, fu);
        } else {
            if (u < x) {
                a = u;
            } else {
                b = u;
            }
            if (fu <= fw || w == x) {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }
    *xmin = x;
    return fx;
}

void mnbrak(
    double* ax,
    double* bx,
    double* cx,
    double* fa,
    double* fb,
    double* fc,
    double (*func)(double, void*),
    void* params
) {
    /*
    Given a function func, and given distinct initial points ax and bx, this routine searches in
    the downhill direction (defined by the function as evaluated at the initial points) and returns
    new points ax, bx, cx that bracket a minimum
     */
    double ulim, u, r, q, fu, dum;
    *fa = (*func)(*ax, params);
    *fb = (*func)(*bx, params);
    if (*fb > *fa) {
        SHFT(dum, *ax, *bx, dum)
        SHFT(dum, *fb, *fa, dum)
    }
    *cx = (*bx) + GOLD * (*bx - *ax);
    *fc = (*func)(*cx, params);
    while (*fb > *fc) {
        r = (*bx - *ax) * (*fb - *fc);
        q = (*bx - *cx) * (*fb - *fa);
        u = (*bx) - ((*bx - *cx) * q - (*bx - *ax) * r) / (2.0 * SIGN(FMAX(fabs(q - r), TINY), q - r));
        ulim = (*bx) + GLIMIT * (*cx - *bx);
        if ((*bx - u) * (u - *cx) > 0.0) {
            fu = (*func)(u, params);
            if (fu < *fc) {
                *ax = (*bx);
                *bx = u;
                *fa = (*fb);
                *fb = fu;
                return;
            } else if (fu > *fb) {
                *cx = u;
                *fc = fu;
                return;
            }
            u = (*cx) + GOLD * (*cx - *bx);
            fu = (*func)(u, params);
        } else if ((*cx - u) * (u - ulim) > 0.0) {
            fu = (*func)(u, params);
            if (fu < *fc) {
                SHFT(*bx, *cx, u, *cx + GOLD * (*cx - *bx))
                SHFT(*fb, *fc, fu, (*func)(u, params))
            }
        } else if ((u - ulim) * (ulim - *cx) >= 0.0) {
            u = ulim;
            fu = (*func)(u, params);
        } else {
            u = (*cx) + GOLD * (*cx - *bx);
            fu = (*func)(u, params);
        }
        SHFT(*ax, *bx, *cx, u)
        SHFT(*fa, *fb, *fc, fu)
    }
}

double minimize_scalar(double start_a, double start_b, double (*func)(double, void*), void* params, double* xmin) {
    double ax = start_a;
    double bx = start_b;
    double cx;  // Will be generated by mnbrak

    // Variables to hold the function values evaluated at the triplet points
    double fa, fb, fc;

    // expand/locate a valid 3-point bracket
    mnbrak(&ax, &bx, &cx, &fa, &fb, &fc, func, params);

    // Find the minimum using the 3 points
    double min_val = brent(ax, bx, cx, func, params, xmin);

    return min_val;
}
