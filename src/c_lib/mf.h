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


int fgmf_old(
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

int fgmf(
    float* tx,                 // 1
    int tx_len,                // 2
    float* rx,                 // 3
    int rx_len,                // 4
    int sub_res_len,           // 5
    float* acc_phasors,        // 6
    int n_accs,                // 7
    double* accelerations,     // 8
    int* rgs,                  // 9
    int n_rg,                  // 10
    int frequency_decimation,  // 11
    double* vals,              // 12
    double* dc,                // 13
    double* v,                 // 14
    double* a,                 // 15
    double* phi,               // 16
    int* rx_window,            // 17
    int* dec_rx_inds,          // 18
    int dec_signal_len,        // 19
    double* fft_frequencies,   // 20
    int fft_frequencies_len,   // 21
    float sample_rate,         // 22
    int refine_acceleration,   // 23
    int refine_doppler         // 24
);


int fdpt(
    float* tx,                 // 1
    int tx_len,                // 2
    float* rx,                 // 3
    int rx_len,                // 4
    int sub_res_len,           // 5
    float* acc_phasors,        // 6
    double* accelerations,     // 7
    int* rgs,                  // 8
    int n_rg,                  // 9
    int frequency_decimation,  // 10
    double* vals,              // 11
    double* dc,                // 12
    double* v,                 // 13
    double* a,                 // 14
    double* phi,               // 15
    int* rx_window,            // 16
    int* dec_rx_inds,          // 17
    int dec_signal_len,        // 18
    int dec_tau_samp,          // 19
    double* fft_frequencies,   // 20
    int fft_frequencies_len,   // 21
    float sample_rate,         // 22
    int refine_acceleration,   // 23
    int refine_doppler         // 24
);

int xcorr_echo_search(
    float complex* tx,                   // 1 input
    int tx_len,                          // 2 size
    float complex* rx,                   // 3 input
    int rx_len,                          // 4 size
    int* doppler_frequencies,            // 5 input
    int doppler_frequencies_len,         // 6 size
    int range_gate_step,                 // 7 input
    int t_samp_usec,                     // 8 input
    float complex* pows,                 // 9 output
    int* pows_size,                      // 10 size
    float complex* pows_normalized,      // 11 output
    int* pows_normalized_size,           // 12 size
    float complex* max_pow_per_doppler,  // 13 output
    int max_pow_per_doppler_size,        // 14 size
    int* max_pow_ind,                    // 15 output
    int max_pow_ind_size                 // 16 size
);
