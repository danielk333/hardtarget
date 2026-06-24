#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mf.h"
#include "optimize/solvers.h"
#include "utils/fftw_utils.h"

/*
  Range-Velocity-Acceleration matched filter

  todo optimizations:
    - avx
    - range dependent acceleration grid. The expected acceleration is a function
      of altitude. we only would need to search through a finite grid around
      the expected value. This would save a lot of computation.

  Notes:
    - Here the input signals are complex but interpreted as floats making
      them 2*len long and interpreted as [ind0_re, ind0_im, ind0_re...]

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
            for (int fi = 0; fi < dec_signal_len; fi++) {
                dec_signal[fi] = 0.0 + 0.0 * I;
            }
            compute_echo_signal(
                echo, echo_len, tx, tx_len, rx, rx_len, sri, sub_res_len, frequency_decimation, rgs[ri], rx_window
            );

            int v_ind = -1;
            int a_ind = -1;

            // for all accelerations
            // add range gate dependent accelerations
            for (int ai = 0; ai < n_accs; ai++) {
                int phasor_i = 2 * ai * echo_len;

                multiply_acc_phasors(dec_signal, echo, echo_len, acc_phasors, phasor_i, dec_rx_inds);

                // execute fft in and store result in out
                fftwf_execute(p);
                // Shift ft
                fft_shift_1d(ft, dec_signal_len);

                float pwr;
                for (int ti = 0; ti < dec_signal_len; ti++) {
                    pwr = cpowf(cabsf(ft[ti]), 2);
                    if (ai == 0 && ti == 0) {
                        // zero-frequency (DC) component in FFTW out[0] according to docs
                        dc[ind] = pwr;
                    }
                    if (pwr > vals[ind]) {
                        vals[ind] = pwr;
                        v_ind = ti;
                        v[ind] = fft_frequencies[ti];
                        a_ind = ai;
                        a[ind] = accelerations[ai];
                        phi[ind] = cargf(ft[ti]);
                    }
                }
            }

            multiply_acc_phasors(dec_signal, echo, echo_len, acc_phasors, 2 * a_ind * echo_len, dec_rx_inds);

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
                float pwr = 0;
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
