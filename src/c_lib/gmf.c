#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mf.h"
#include "optimize/solvers.h"
#include "utils/fftw_utils.h"

/*
  Range-Velocity-Acceleration matched filter
  The commented numbers are the argument numbers, useful for debugging the ctypes interface.
 */

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
) {
    fftwf_complex* echo;
    fftwf_complex* dec_signal;
    fftwf_complex* ft;

    fftwf_plan p;
    int echo_len;

    echo_len = (int)(tx_len / frequency_decimation);
    echo = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * echo_len);
    dec_signal = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_signal_len);
    ft = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_signal_len);

    p = fftwf_plan_dft_1d(dec_signal_len, dec_signal, ft, FFTW_FORWARD, FFT_PLAN_ID);

    // for each range gate
    for (int ri = 0; ri < n_rg; ri++) {
        // For each sub resolution
        for (int sri = 0; sri < sub_res_len; sri++) {
            int ind = sri + ri * sub_res_len;

            // Reset dec_signal
            memset(dec_signal, 0, sizeof(*dec_signal) * dec_signal_len);

            compute_echo_signal(
                echo, echo_len, tx, tx_len, rx, rx_len, sri, sub_res_len, frequency_decimation, rgs[ri], rx_window
            );

            float complex echo_sum = 0.0f + 0.0f * I;

            for (int i = 0; i < echo_len; i++) {
                echo_sum += echo[i];
            }

            float real = crealf(echo_sum);
            float imag = cimagf(echo_sum);
            dc[ind] = real * real + imag * imag;

            // for all accelerations
            // add range gate dependent accelerations
            int v_ind = -1;
            int a_ind = -1;
            float best_real = 0.0f;
            float best_imag = 0.0f;
            for (int ai = 0; ai < n_accs; ai++) {
                int phasor_i = 2 * ai * echo_len;

                multiply_acc_phasors(dec_signal, echo, echo_len, acc_phasors, phasor_i, dec_rx_inds);

                // execute fft in and store result in out
                fftwf_execute(p);
                // Shift ft
                fft_shift_1d(ft, dec_signal_len);

                for (int ti = 0; ti < dec_signal_len; ti++) {
                    float real = crealf(ft[ti]);
                    float imag = cimagf(ft[ti]);
                    float pwr = (real * real) + (imag * imag);
                    if (pwr > vals[ind]) {
                        vals[ind] = pwr;
                        v_ind = ti;
                        a_ind = ai;
                        best_real = real;
                        best_imag = imag;
                    }
                }
            }

            // Store best results
            v[ind] = fft_frequencies[v_ind];
            a[ind] = accelerations[a_ind];
            phi[ind] = atan2f(best_imag, best_real);

            int v_ind_p = v_ind;
            if (v_ind < fft_frequencies_len - 1) {
                v_ind_p += 1;
            }
            int v_ind_m = v_ind;
            if (v_ind > 0) {
                v_ind_m -= 1;
            }

            if (refine_acceleration) {
                // do stuff
            } else if (refine_doppler) {
                double pwr = 0;

                v[ind] = dtft_solve(
                    dec_signal,
                    dec_signal_len,
                    sample_rate,
                    fft_frequencies[v_ind_m],
                    fft_frequencies[v_ind_p],
                    &pwr,
                    &phi[ind]
                );
            }
        }
    }
    fftwf_free(dec_signal);
    fftwf_free(ft);
    fftwf_free(echo);
    fftwf_destroy_plan(p);
    return 0;  // Success
}
