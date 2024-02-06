#include "gmf.h"

#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

#define ECHO 1
#define PEAK_SEARCH 1
#define FFT_VECTOR 1
#define ACC_MULT 1
#define FFT_PLAN_ID FFTW_MEASURE

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


  Range-Velocity-Acceleration matched filter

  todo optimizations:
    - avx
    - range dependent acceleration grid. The expected acceleration is a function
      of altitude. we only would need to search through a finite grid around
      the expected value. This would save a lot of computation.

  Notes:
    - Here the input signals are complex but interpreted as floats making 
      them 2*len long and interpreted as [ind0_re, ind0_im, ind0_re...]

 */
int gmf(
    float *z_tx, int z_tx_len, float *z_rx, int z_rx_len,
    float *acc_phasors, int n_accs, int *rgs, int n_rg,
    int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, 
    int *a_vec, int *rx_window, int *dec_rx_inds, int dec_signal_len
) {
    fftwf_complex *echo;
    fftwf_complex *in;
    fftwf_complex *out;

    fftwf_plan p;
    int echo_len;

    echo_len = (int)(z_tx_len / dec);
    echo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*echo_len);
    in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*dec_signal_len);
    out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*dec_signal_len);

    p = fftwf_plan_dft_1d(dec_signal_len, in, out, FFTW_FORWARD, FFT_PLAN_ID);

    // for each range gate
    for (int ri = 0; ri < n_rg; ri++) {
#ifdef ECHO
        // zero echo
        for (int fi = 0; fi < echo_len; fi++) {
            echo[fi][0] = 0.0;
            echo[fi][1] = 0.0;
        }
        for (int fi = 0; fi < dec_signal_len; fi++) {
            in[fi][0] = 0.0;
            in[fi][1] = 0.0;
        }
        int tidx;
        for (int ti = 0; ti < z_tx_len; ti++) {
            // rea*reb - ima*imb
            // z_tx*conj(z_rx)
            tidx = ti / dec;
            // Real part of z_t[ti]x*z_rx[rg+ti]
            echo[tidx][0] += z_tx[2*ti]*z_rx[(rx_window[ti] + rgs[ri])*2] - z_tx[2*ti + 1]*z_rx[(rx_window[ti] + rgs[ri])*2 + 1];
            // rea*imb + ima*reb
            // Imag part of z_t[ti]x*z_rx[rg+ti]
            echo[tidx][1] += z_tx[2*ti]*z_rx[(rx_window[ti] + rgs[ri])*2 + 1] + z_tx[2*ti + 1]*z_rx[(rx_window[ti] + rgs[ri])*2];
        }
#endif  // ECHO
        // for all accelerations
        // add range gate dependent accelerations
        for (int ai = 0; ai < n_accs; ai++) {
#ifdef ACC_MULT
            int phasor_i = 2*ai*echo_len;
            // echo*acc_phasors
            float rep, imp;
            for (int tidx = 0; tidx < echo_len; tidx++) {
                rep = acc_phasors[phasor_i + 2*tidx];
                imp = acc_phasors[phasor_i + 2*tidx + 1];

                // rea*reb - ima*imb
                in[dec_rx_inds[tidx]][0] = echo[tidx][0]*rep - echo[tidx][1]*imp;
                // rea*imb + ima*reb
                in[dec_rx_inds[tidx]][1] = echo[tidx][0]*imp + echo[tidx][1]*rep;
            }
#endif
#ifdef FFT_VECTOR
            // fft in and store result in out
            fftwf_execute(p);
#endif
#ifdef PEAK_SEARCH
            float gmf2;
            for (int ti = 0; ti < dec_signal_len; ti++) {
                gmf2 = out[ti][0]*out[ti][0] + out[ti][1]*out[ti][1];
                if (ai == 0 && ti == 0) {
                    // zero-frequency (DC) component in FFTW out[0] according to docs
                    gmf_dc_vec[ri] = gmf2;
                }
                if (gmf2 > gmf_vec[ri]) {
                    gmf_vec[ri] = gmf2;
                    v_vec[ri] = ti;  // frequency index
                    a_vec[ri] = ai;  // acceleration index
                }
            }
#endif
        }
    }
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(echo);
    fftwf_destroy_plan(p);
    return 0; // Success
}
