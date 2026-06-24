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

float complex dtft_fractor_sum(float freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len) {
    // Accumulate the DTFT sum using native single-precision float complex types
    float complex sum = 0.0f + 0.0f * I;

    for (int n = 0; n < signal_len; n++) {
        double t = (float)n / sample_rate;

        float complex dtft_fractor = cexp(2.0 * M_PI * freq * t * -I);

        sum += dtft_fractor * decoded_signal[n];
    }
    return sum;
}

float dtft_optimize_fun(float freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len) {
    float complex fractor_sum = dtft_fractor_sum(freq, decoded_signal, sample_rate, signal_len);

    // Absolute magnitude
    float mag = cabsf(fractor_sum);
    // Return negative magnitude squared (converting maximization to minimization)
    return -(float)(mag * mag);
}

double dtft_solve(
    const fftwf_complex* decoded_signal,
    int signal_len,
    float sample_rate,
    float freq_start,
    float freq_end,
    float* fmin_pwr,
    double* phi
) {
    double a = freq_start;
    double b = freq_end;
    double freq_est = 0.0;
    double value = 0.0;
    int status = 0;

    while (1) {
        status = local_min_rc(&a, &b, &status, value);

        if (status <= 0) {
            // IF minimization completed or converged, break loop.
            break;
        }

        value = dtft_optimize_fun(freq_est, decoded_signal, sample_rate, signal_len);
    }

    // Power
    *fmin_pwr = -value;

    // Phase estimation
    float complex fractor_sum = dtft_fractor_sum(freq_est, decoded_signal, sample_rate, signal_len);
    *phi = cargf(fractor_sum);

    return freq_est;  // optimized freq
}
