#include <fftw3.h> // Note, fftw3 must be first to avoid conflicts with complex.h
#include <complex.h>
#include <math.h>

/*
Choosing FFTW plan flag, citing [fftw.org docs](https://www.fftw.org/fftw3_doc/Planner-Flags.html):
 - FFTW_ESTIMATE specifies that, instead of actual measurements of different algorithms, a simple
    heuristic is used to pick a (probably sub-optimal) plan quickly. With this flag, the
    input/output arrays are not overwritten during planning.
 - FFTW_MEASURE tells FFTW to find an optimized plan by actually computing several FFTs and
    measuring their execution time. Depending on your machine, this can take some time (often a few
    seconds). FFTW_MEASURE is the default planning option.
 - FFTW_PATIENT is like FFTW_MEASURE, but considers a wider range of algorithms and often produces
    a “more optimal” plan (especially for large transforms), but at the expense of several times
    longer planning time (especially for large transforms).
 - FFTW_EXHAUSTIVE is like FFTW_PATIENT, but considers an even wider range of algorithms, including
    many that we think are unlikely to be fast, to produce the most optimal plan but with a
    substantially increased planning time.
*/

#define FFT_PLAN_ID FFTW_MEASURE


int fgmf(
    float* tx,
    int tx_len,
    float* rx,
    int rx_len,
    int sub_res_len,
    float* acc_phasors,
    int n_accs,
    int* rgs,
    int n_rg,
    int dec,
    float* gmf_vec,
    float* gmf_dc_vec,
    int* v_vec,
    int* a_vec,
    int* rx_window,
    int* dec_rx_inds,
    int dec_signal_len
);

int fdpt(
    float* tx,
    int tx_len,
    float* rx,
    int rx_len,
    int sub_res_len,
    float* acc_phasors,
    int n_accs,
    int* rgs,
    int n_rg,
    int dec,
    float* gmf_vec,
    float* gmf_dc_vec,
    int* v_vec,
    int* a_vec,
    int* rx_window,
    int* dec_rx_inds,
    int dec_signal_len,
    int dec_tau_samp
);

int xcorr_echo_search(
    float complex* tx,
    int tx_len,
    float complex* rx,
    int rx_len,
    int doppler_freq_min,
    int doppler_freq_max,
    int doppler_freq_step,
    int doppler_freq_size,
    int t_samp_usec,
    float complex* pows,
    int* pows_size,
    float complex* pows_normalized,
    int* pows_normalized_size,
    float complex* max_pow_per_doppler,
    int max_pow_per_doppler_size,
    int* max_pow_ind,
    int max_pow_ind_size
);
