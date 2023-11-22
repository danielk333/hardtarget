#include "gmf.h"

#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

#define ECHO 1
#define PEAK_SEARCH 1
#define FFT_VECTOR 1
#define ACC_MULT 1
/*
  Range-Velocity-Acceleration matched filter

  todo optimizations:
  - avx
  - range dependent acceleration grid. The expected acceleration is a function
    of altitude. we only would need to search through a finite grid around
    the expected value. This would save a lot of computation.

    Here the input signals are complex but interpreted as floats making 
    them 2*len long and interpreted as [ind0_re, ind0_im, ind0_re...]

 */
int gmf(float *z_tx, int z_tx_len, float *z_rx, int z_rx_len, float *acc_phasors, int n_accs, int *rgs, int n_rg,
        int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, int *a_vec, int *rx_window) {
    fftwf_complex *echo;
    fftwf_complex *in;
    fftwf_complex *out;

    fftwf_plan p;
    int rg;
    int nfft2;

    float *rx_select;
    rx_select = (float *)malloc(sizeof(fftwf_complex)*z_tx_len*2);

    nfft2 = (int)(z_tx_len / dec);
    echo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nfft2);
    in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nfft2);
    out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nfft2);

    // p = fftwf_plan_dft_1d(nfft2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    p = fftwf_plan_dft_1d(nfft2, in, out, FFTW_FORWARD, FFTW_MEASURE);

    //
    // in order to reduce the number of operations later on, determine the
    // number of non-zero transmit pulses. we don't need to calculate
    // zero values.
    //
    float *tx_power;
    float *tx_power2;
    tx_power = (float *)malloc(sizeof(float)*z_tx_len);
    tx_power2 = (float *)malloc(sizeof(float)*nfft2);

    // zero tx power
    for (int ti = 0; ti < z_tx_len; ti++) {
        tx_power[ti] = 0.0;
    }
    for (int ti = 0; ti < nfft2; ti++) {
        tx_power2[ti] = 0.0;
    }

    int n_nonzero_tx = 0;
    for (int ti = 0; ti < z_tx_len; ti++) {
        tx_power[ti] = z_tx[2*ti]*z_tx[2*ti] + z_tx[2*ti + 1]*z_tx[2*ti + 1];
        tx_power2[ti / dec] += z_tx[2*ti]*z_tx[2*ti] + z_tx[2*ti + 1]*z_tx[2*ti + 1];
        if (tx_power[ti] > 1e-10) n_nonzero_tx++;
    }
    int n_nonzero_tx2 = 0;
    for (int ti = 0; ti < nfft2; ti++) {
        if (tx_power2[ti] > 1e-10) n_nonzero_tx2++;
    }

    int *tx_idx;
    tx_idx = malloc(sizeof(int)*n_nonzero_tx);
    //  n_nonzero_tx_dec=n_nonzero_tx/dec;
    int nzi = 0;
    for (int ti = 0; ti < z_tx_len; ti++) {
        if (tx_power[ti] > 1e-10) {
            tx_idx[nzi] = ti;
            nzi++;
        }
    }
    // What elements for the decimated echo*tx vector are non-zero
    int *tx_idx2;
    tx_idx2 = malloc(sizeof(int)*n_nonzero_tx2);
    nzi = 0;
    for (int ti = 0; ti < nfft2; ti++) {
        if (tx_power2[ti] > 1e-10) {
            tx_idx2[nzi] = ti;
            nzi++;
        }
    }

    // for each range gate
    for (int ri = 0; ri < n_rg; ri++) {
#ifdef ECHO
        // zero echo
        for (int fi = 0; fi < nfft2; fi++) {
            echo[fi][0] = 0.0;
            echo[fi][1] = 0.0;
            in[fi][0] = 0.0;
            in[fi][1] = 0.0;
        }
        int rg = rgs[ri];
        int rxwi;
        for (int rxi = 0; rxi < z_tx_len*2; rxi++) {
            rxwi = (int)((float)rxi / 2.0);
            rx_select[rxi] = z_rx[rx_window[rxwi] + rg];
        }

        for (int ni = 0; ni < n_nonzero_tx; ni++) {
            int ti = tx_idx[ni];
            //                rea*reb        -ima*imb
            // z_tx*conj(z_rx)
            int tidx = ti / dec;
            // Real part of z_t[ti]x*z_rx[rg+ti]
            echo[tidx][0] += z_tx[2*ti]*rx_select[2*ti] - z_tx[2*ti + 1]*rx_select[2*ti + 1];
            //                rea*imb        + ima*reb
            // Imag part of z_t[ti]x*z_rx[rg+ti]
            echo[tidx][1] += z_tx[2*ti]*rx_select[2*ti + 1] + z_tx[2*ti + 1]*rx_select[2*ti];
        }
#endif  // ECHO
        // for all accelerations
        // add range gate dependent accelerations
        for (int ai = 0; ai < n_accs; ai++) {
            int phasor_i = 2*ai*nfft2;
            // echo*acc_phasors
            // only multiply what is needed
#ifdef ACC_MULT
            for (int ni = 0; ni < n_nonzero_tx2; ni++) {
                int ti = tx_idx2[ni];
                float rep = acc_phasors[phasor_i + 2*ti];
                float imp = acc_phasors[phasor_i + 2*ti + 1];

                // rea*reb - ima*imb
                in[ti][0] = echo[ti][0]*rep - echo[ti][1]*imp;
                // rea*imb + ima*reb
                in[ti][1] = echo[ti][0]*imp + echo[ti][1]*rep;
            }
#endif
            // fft in and store result in out
#ifdef FFT_VECTOR
            fftwf_execute(p);
#endif
#ifdef PEAK_SEARCH
            float gmf2;
            for (int ti = 0; ti < nfft2; ti++) {
                gmf2 = out[ti][0]*out[ti][0] + out[ti][1]*out[ti][1];
                if (ai == 0 && ti == 0) {
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
    free(tx_power);
    free(tx_power2);
    free(tx_idx);
    free(tx_idx2);
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(echo);
    fftwf_destroy_plan(p);
    return 0; // Success
}
