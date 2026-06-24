
#include <complex.h>
#include <fftw3.h>


float complex dtft_fractor_sum(float freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len);
float dtft_optimize_fun(float freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len);
float dtft_solve(
    const fftwf_complex* decoded_signal,
    int signal_len,
    float sample_rate,
    float freq_start,
    float freq_end,
    float* fmin_pwr,
    double* phi
) ;
