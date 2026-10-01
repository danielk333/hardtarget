#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Include complex.h first, this binds fftwf_complex directly to native C99 'float complex' types.
#include <complex.h>
#include <fftw3.h>

#include "brent.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double complex dtft_fractor_sum(double freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len) {
    // Accumulate the DTFT sum using native single-precision float complex types
    double complex sum = 0.0 + 0.0 * I;

    double delta_theta = -2.0 * M_PI * freq / (double)sample_rate;
    double complex rotation_step = cos(delta_theta) + sin(delta_theta) * I;

    // Start for  e^(i*0) which is 1.0
    double complex dtft_fractor = 1.0 + 0.0 * I;

    for (int n = 0; n < signal_len; n++) {
        sum += dtft_fractor * (decoded_signal[n]);
        dtft_fractor *= rotation_step;
    }
    return sum;
}

double dtft_optimize_fun(double freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len) {
    double complex fractor_sum = dtft_fractor_sum(freq, decoded_signal, sample_rate, signal_len);

    // Absolute magnitude
    double real = creal(fractor_sum);
    double imag = cimag(fractor_sum);
    double mag = (real * real) + (imag * imag);  // Faster variant of cabs

    // Return negative magnitude squared (converting maximization to minimization)
    return -mag;
}

typedef struct {
    const fftwf_complex* decoded_signal;
    float sample_rate;
    int signal_len;
} dtft_params;

static double dtft_optimize_wrapper(double freq, void* params) {
    dtft_params* ctx = (dtft_params*)params;

    return dtft_optimize_fun(freq, ctx->decoded_signal, ctx->sample_rate, ctx->signal_len);
}
double dtft_solve(
    const fftwf_complex* decoded_signal,
    int signal_len,
    float sample_rate,
    double freq_start,
    double freq_end,
    double* fmin_pwr,
    double* phi
) {
    double freq_est = 0.0;
    double value = 0.0;

    dtft_params params = {.decoded_signal = decoded_signal, .sample_rate = sample_rate, .signal_len = signal_len};

    value = minimize_scalar(freq_start, freq_end, &dtft_optimize_wrapper, &params, &freq_est);

    // Power
    *fmin_pwr = -value;

    // Phase estimation
    double complex fractor_sum = dtft_fractor_sum(freq_est, decoded_signal, sample_rate, signal_len);

    *phi = atan2(cimag(fractor_sum), creal(fractor_sum));

    return freq_est;  // optimized freq
}
