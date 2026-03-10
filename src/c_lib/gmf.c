#include "mf.h"
#include "utils/fftw_utils.h"
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>


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
    float* tx,         // 1
    int tx_len,        // 2
    float* rx,         // 3
    int rx_len,        // 4
    int sub_res_len,     // 5
    float* acc_phasors,  // 6
    int n_accs,          // 7
    int* rgs,            // 8
    int n_rg,            // 9
    int dec,             // 10
    float* gmf_vec,      // 11
    float* gmf_dc_vec,   // 12
    int* v_vec,          // 13
    int* a_vec,          // 14
    int* rx_window,      // 15
    int* dec_rx_inds,    // 16
    int dec_signal_len   // 17
) {
    fftwf_complex* echo;
    fftwf_complex* in;
    fftwf_complex* out;

    fftwf_plan p;
    int echo_len;

    echo_len = (int)(tx_len / dec);
    echo = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * echo_len);
    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_signal_len);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_signal_len);

    p = fftwf_plan_dft_1d(dec_signal_len, in, out, FFTW_FORWARD, FFT_PLAN_ID);

    // for each range gate
    for (int ri = 0; ri < n_rg; ri++) {
        for (int sri = 0; sri < sub_res_len; sri++) {
            int ind = sri + ri * sub_res_len;
            for (int fi = 0; fi < dec_signal_len; fi++) {
                in[fi][0] = 0.0;
                in[fi][1] = 0.0;
            }
            compute_echo_signal(
                echo, echo_len, tx, tx_len, rx, rx_len, sri, sub_res_len, dec, rgs[ri], rx_window
            );

            // for all accelerations
            // add range gate dependent accelerations
            for (int ai = 0; ai < n_accs; ai++) {
                int phasor_i = 2 * ai * echo_len;

                multiply_acc_phasors(in, echo, echo_len, acc_phasors, phasor_i, dec_rx_inds);

                // fft in and store result in out
                fftwf_execute(p);

                float gmf2;
                for (int ti = 0; ti < dec_signal_len; ti++) {
                    gmf2 = out[ti][0] * out[ti][0] + out[ti][1] * out[ti][1];
                    if (ai == 0 && ti == 0) {
                        // zero-frequency (DC) component in FFTW out[0] according to docs
                        gmf_dc_vec[ind] = gmf2;
                    }
                    if (gmf2 > gmf_vec[ind]) {
                        gmf_vec[ind] = gmf2;
                        v_vec[ind] = ti;  // frequency index
                        a_vec[ind] = ai;  // acceleration index
                    }
                }
            }
        }
    }
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(echo);
    fftwf_destroy_plan(p);
    return 0;  // Success
}
