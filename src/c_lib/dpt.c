#include "mf.h"
#include "utils/fftw_utils.h"
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

/*
Fast Discrete Polynomial Phase transform

TODO: this can be quite optimized by removing unnessary operations
*/
int fdpt(
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
    int dec_signal_len,  // 17
    int dec_tau_samp     // 18
) {
    fftwf_complex* echo;
    fftwf_complex* in;
    fftwf_complex* in_tau;
    fftwf_complex* out;
    fftwf_complex* out_tau;

    fftwf_plan p;
    fftwf_plan p_tau;
    int echo_len;

    echo_len = (int)(tx_len / dec);
    echo = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * echo_len);
    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_signal_len);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_signal_len);
    in_tau = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_tau_samp);
    out_tau = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dec_tau_samp);

    p = fftwf_plan_dft_1d(dec_signal_len, in, out, FFTW_FORWARD, FFT_PLAN_ID);
    p_tau = fftwf_plan_dft_1d(dec_tau_samp, in_tau, out_tau, FFTW_FORWARD, FFT_PLAN_ID);
    int in_tau_peak;
    int in_peak;
    int phasor_i;

    // for each range gate
    for (int ri = 0; ri < n_rg; ri++) {
        for (int sri = 0; sri < sub_res_len; sri++) {
            int ind = sri + ri * sub_res_len;
            for (int fi = 0; fi < dec_signal_len; fi++) {
                in[fi][0] = 0.0;
                in[fi][1] = 0.0;
            }
            for (int fi = 0; fi < dec_tau_samp; fi++) {
                in_tau[fi][0] = 0.0;
                in_tau[fi][1] = 0.0;
            }
            compute_echo_signal(
                echo, echo_len, tx, tx_len, rx, rx_len, sri, sub_res_len, dec, rgs[ri], rx_window
            );
            gmf_dc_vec[ind] = compute_echo_power(echo, echo_len);

            compute_phase_difference(in, dec_signal_len, in_tau, dec_tau_samp, echo, echo_len, dec_rx_inds);
            fftwf_execute(p_tau);

            in_tau_peak = find_fftwf_peak(out_tau, dec_tau_samp);

            // FFT shift the tau peak
            in_tau_peak = (in_tau_peak + dec_tau_samp / 2) % dec_tau_samp;
            // Calculate phasor location in array
            phasor_i = 2 * in_tau_peak * echo_len;

            multiply_acc_phasors(in, echo, echo_len, acc_phasors, phasor_i, dec_rx_inds);

            // fft in and store result in out
            fftwf_execute(p);

            in_peak = find_fftwf_peak(out, dec_signal_len);

            gmf_vec[ind] = out[in_peak][0] * out[in_peak][0] + out[in_peak][1] * out[in_peak][1];
            v_vec[ind] = in_peak;      // frequency index
            a_vec[ind] = in_tau_peak;  // acceleration index
        }
    }
    fftwf_free(in);
    fftwf_free(out);
    fftwf_free(in_tau);
    fftwf_free(out_tau);
    fftwf_free(echo);
    fftwf_destroy_plan(p);
    fftwf_destroy_plan(p_tau);
    return 0;  // Success
}
