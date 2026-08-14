
#include <complex.h>
#include <fftw3.h>


double complex dtft_fractor_sum(double freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len);
double dtft_optimize_fun(double freq, const fftwf_complex* decoded_signal, float sample_rate, int signal_len);
double dtft_solve(
    const fftwf_complex* decoded_signal,
    int signal_len,
    float sample_rate,
    double freq_start,
    double freq_end,
    double* fmin_pwr,
    double* phi
) ;
