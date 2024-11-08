#include "gmf.h"

#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

#define FFT_PLAN_ID FFTW_MEASURE


void compute_echo_signal(
    fftwf_complex *echo, int echo_len,
    float *z_tx, int z_tx_len,
    float *z_rx, int z_rx_len,
    int decimation,
    int rg,
    int *rx_window
) {
    // zero echo
    for (int fi = 0; fi < echo_len; fi++) {
        echo[fi][0] = 0.0;
        echo[fi][1] = 0.0;
    }
    int tidx;
    for (int ti = 0; ti < z_tx_len; ti++) {
        // rea*reb - ima*imb
        // z_tx*conj(z_rx)
        tidx = ti / decimation;
        // Real part of z_t[ti]x*z_rx[rg+ti]
        echo[tidx][0] += z_tx[2*ti]*z_rx[(rx_window[ti] + rg)*2] - z_tx[2*ti + 1]*z_rx[(rx_window[ti] + rg)*2 + 1];
        // rea*imb + ima*reb
        // Imag part of z_t[ti]x*z_rx[rg+ti]
        echo[tidx][1] += z_tx[2*ti]*z_rx[(rx_window[ti] + rg)*2 + 1] + z_tx[2*ti + 1]*z_rx[(rx_window[ti] + rg)*2];
    }
}

float compute_echo_power(
    fftwf_complex *echo, int echo_len
) {
    float sum_re = 0;
    float sum_im = 0;
    for (int i = 0; i < echo_len; i++) {
        sum_re += echo[i][0];
        sum_im += echo[i][1];
    }
    return sum_re*sum_re + sum_im*sum_im;
}

void multiply_acc_phasors(
    fftwf_complex *in,
    fftwf_complex *echo, int echo_len,
    float *acc_phasors,
    int phasor_index,
    int *dec_rx_inds
) {
    // echo*acc_phasors
    float rep, imp;
    for (int tidx = 0; tidx < echo_len; tidx++) {
        rep = acc_phasors[phasor_index + 2*tidx];
        imp = acc_phasors[phasor_index + 2*tidx + 1];

        // rea*reb - ima*imb
        in[dec_rx_inds[tidx]][0] = echo[tidx][0]*rep - echo[tidx][1]*imp;
        // rea*imb + ima*reb
        in[dec_rx_inds[tidx]][1] = echo[tidx][0]*imp + echo[tidx][1]*rep;
    }
}

void compute_phase_difference(
    fftwf_complex *in, int dec_signal_len,
    fftwf_complex *in_tau, int dec_tau_samp,
    fftwf_complex *echo, int echo_len,
    int *dec_rx_inds
) {
    for (int tidx = 0; tidx < echo_len; tidx++) {
        in[dec_rx_inds[tidx]][0] = echo[tidx][0];
        in[dec_rx_inds[tidx]][1] = echo[tidx][1];
    }
    int ti_inv;
    for (int ti = 0; ti < dec_tau_samp; ti++) {
        ti_inv = dec_tau_samp + ti;
        // dec_signal[dec_tau_samp:] * np.conj(dec_signal[:-dec_tau_samp])
        // formula for complex mult
        // rea*reb - ima*imb
        in_tau[ti][0] = in[ti_inv][0] * in[ti][0] + in[ti_inv][1] * in[ti][1];
        // rea*imb + ima*reb
        in_tau[ti][1] = -in[ti_inv][0] * in[ti][1] + in[ti_inv][1] * in[ti][0];
        // But we take imb = -imb for the complex conj
    }
}

int find_fftwf_peak(fftwf_complex *arr, int len) {
    int index = 0;
    float abs_val;
    float max_val = 0;
    for (int i = 0; i < len; i++) {
        abs_val = arr[i][0]*arr[i][0] + arr[i][1]*arr[i][1];
        if (abs_val > max_val) {
            max_val = abs_val;
            index = i;
        }
    }
    return index;
}

/*
Fast Discrete Polynomial Phase transform

TODO: this can be quite optimized by removing unnessary operations
*/
int dpt(
    float *z_tx, int z_tx_len, // 1, 2
    float *z_rx, int z_rx_len, // 3, 4
    float *acc_phasors, int n_accs, // 5, 6
    int *rgs, int n_rg, // 7, 8
    int dec, // 9
    float *gmf_vec, // 10
    float *gmf_dc_vec, // 11
    int *v_vec, // 12
    int *a_vec, // 13
    int *rx_window, // 14
    int *dec_rx_inds, // 15
    int dec_signal_len, // 16
    int dec_tau_samp // 17
) {
    fftwf_complex *echo;
    fftwf_complex *in;
    fftwf_complex *in_tau;
    fftwf_complex *out;
    fftwf_complex *out_tau;

    fftwf_plan p;
    fftwf_plan p_tau;
    int echo_len;

    echo_len = (int)(z_tx_len / dec);
    echo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*echo_len);
    in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*dec_signal_len);
    out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*dec_signal_len);
    in_tau = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*dec_tau_samp);
    out_tau = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*dec_tau_samp);

    p = fftwf_plan_dft_1d(dec_signal_len, in, out, FFTW_FORWARD, FFT_PLAN_ID);
    p_tau = fftwf_plan_dft_1d(dec_tau_samp, in_tau, out_tau, FFTW_FORWARD, FFT_PLAN_ID);
    int in_tau_peak;
    int in_peak;
    int phasor_i;

    // for each range gate
    for (int ri = 0; ri < n_rg; ri++) {
        for (int fi = 0; fi < dec_signal_len; fi++) {
            in[fi][0] = 0.0;
            in[fi][1] = 0.0;
        }
        for (int fi = 0; fi < dec_tau_samp; fi++) {
            in_tau[fi][0] = 0.0;
            in_tau[fi][1] = 0.0;
        }
        compute_echo_signal(
            echo, echo_len,
            z_tx, z_tx_len,
            z_rx, z_rx_len,
            dec,
            rgs[ri],
            rx_window
        );
        gmf_dc_vec[ri] = compute_echo_power(echo, echo_len);

        compute_phase_difference(
            in, dec_signal_len,
            in_tau, dec_tau_samp,
            echo, echo_len,
            dec_rx_inds
        );
        fftwf_execute(p_tau);

        in_tau_peak = find_fftwf_peak(out_tau, dec_tau_samp);

        //FFT shift the tau peak
        in_tau_peak = (in_tau_peak + dec_tau_samp/2) % dec_tau_samp;
        // Calculate phasor location in array
        phasor_i = 2*in_tau_peak*echo_len;

        multiply_acc_phasors(
            in,
            echo, echo_len,
            acc_phasors,
            phasor_i,
            dec_rx_inds
        );

        // fft in and store result in out
        fftwf_execute(p);

        in_peak = find_fftwf_peak(out, dec_signal_len);

        gmf_vec[ri] = out[in_peak][0]*out[in_peak][0] + out[in_peak][1]*out[in_peak][1];
        v_vec[ri] = in_peak;  // frequency index
        a_vec[ri] = in_tau_peak;  // acceleration index
    }
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(in_tau);
    fftwf_free(out_tau);
    fftwf_free(echo);
    fftwf_destroy_plan(p);
    fftwf_destroy_plan(p_tau);
    return 0; // Success
}

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

    The commented numbers are the argument numbers, useful for debugging the ctypes interface.

 */
int gmf(
    float *z_tx, int z_tx_len, // 1, 2
    float *z_rx, int z_rx_len, // 3, 4
    float *acc_phasors, int n_accs, // 5, 6
    int *rgs, int n_rg, // 7, 8
    int dec, // 9
    float *gmf_vec, // 10
    float *gmf_dc_vec, // 11
    int *v_vec, // 12
    int *a_vec, // 13
    int *rx_window, // 14
    int *dec_rx_inds, // 15
    int dec_signal_len // 16
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
        for (int fi = 0; fi < dec_signal_len; fi++) {
            in[fi][0] = 0.0;
            in[fi][1] = 0.0;
        }
        compute_echo_signal(
            echo, echo_len,
            z_tx, z_tx_len,
            z_rx, z_rx_len,
            dec,
            rgs[ri],
            rx_window
        );

        // for all accelerations
        // add range gate dependent accelerations
        for (int ai = 0; ai < n_accs; ai++) {
            int phasor_i = 2*ai*echo_len;

            multiply_acc_phasors(
                in,
                echo, echo_len,
                acc_phasors,
                phasor_i,
                dec_rx_inds
            );

            // fft in and store result in out
            fftwf_execute(p);

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
        }
    }
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(echo);
    fftwf_destroy_plan(p);
    return 0; // Success
}
