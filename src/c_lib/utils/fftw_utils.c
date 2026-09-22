#include "fftw_utils.h"

#include <fftw3.h>
#include <stdlib.h>
#include <string.h>

// complex arrays are indexed as tx[2*index] = real part, tx[2*index+1] = imaginary part

void compute_echo_signal(
    fftwf_complex* echo,
    int echo_len,
    float* tx,
    int tx_len,
    float* rx,
    int rx_len,
    int sub_res,
    int sub_res_len,
    int decimation,
    int rg,
    int* rx_window
) {
    // Temporary double-precision accumulators.
    double echo_re[echo_len];
    double echo_im[echo_len];

    memset(echo_re, 0, (size_t)echo_len * sizeof(double));
    memset(echo_im, 0, (size_t)echo_len * sizeof(double));

    for (int ti = 0; ti < tx_len; ti++) {
        int tidx = ti / decimation;

        int tx_real_i = 2 * (ti * sub_res_len + sub_res);
        int tx_imag_i = tx_real_i + 1;

        int rx_i = 2 * (rx_window[ti] + rg);

        float tx_re = tx[tx_real_i];
        float tx_im = tx[tx_imag_i];
        float rx_re = rx[rx_i];
        float rx_im = rx[rx_i + 1];

        float p1 = tx_re * rx_re;
        float p2 = tx_im * rx_im;

        // rx * conj(tx)
        float product_re = tx_re * rx_re + tx_im * rx_im;
        float product_im = tx_re * rx_im - tx_im * rx_re;

        // Accumulate in float64 to not accumulate any errors.
        echo_re[tidx] += (double)product_re;
        echo_im[tidx] += (double)product_im;
    }

    // Convert to float32 only once, after all accumulation.
    for (int i = 0; i < echo_len; i++) {
        echo[i][0] = (float)echo_re[i];
        echo[i][1] = (float)echo_im[i];
    }
}

double compute_echo_power(fftwf_complex* echo, int echo_len) {
    double sum_re = 0.0;
    double sum_im = 0.0;

    for (int i = 0; i < echo_len; i++) {
        sum_re += (double)echo[i][0];
        sum_im += (double)echo[i][1];
    }

    return sum_re * sum_re + sum_im * sum_im;
}

void multiply_acc_phasors(
    fftwf_complex* in,
    fftwf_complex* echo,
    int echo_len,
    float* acc_phasors,
    int phasor_index,
    int* dec_rx_inds,
    int drg
) {
    for (int tidx = 0; tidx < echo_len; tidx++) {
        float rep = acc_phasors[phasor_index + 2 * tidx];
        float imp = acc_phasors[phasor_index + 2 * tidx + 1];

        int out_index = dec_rx_inds[tidx] + drg;

        in[out_index][0] = echo[tidx][0] * rep - echo[tidx][1] * imp;

        in[out_index][1] = echo[tidx][0] * imp + echo[tidx][1] * rep;
    }
}

void array_acc_multiply(fftwf_complex* arr1, int arr_len, float* acc_phasors, int phasor_index, fftwf_complex* out) {
    for (int tidx = 0; tidx < arr_len; tidx++) {
        float rep = acc_phasors[phasor_index + 2 * tidx];
        float imp = acc_phasors[phasor_index + 2 * tidx + 1];

        out[tidx][0] = arr1[tidx][0] * rep - arr1[tidx][1] * imp;

        out[tidx][1] = arr1[tidx][0] * imp + arr1[tidx][1] * rep;
    }
}

void compute_phase_difference(
    fftwf_complex* in,
    int dec_signal_len,
    fftwf_complex* in_tau,
    int dec_tau_samp,
    fftwf_complex* echo,
    int echo_len,
    int* dec_rx_inds,
    int drg
) {
    for (int tidx = 0; tidx < echo_len; tidx++) {
        int out_index = dec_rx_inds[tidx] + drg;

        in[out_index][0] = echo[tidx][0];
        in[out_index][1] = echo[tidx][1];
    }

    for (int ti = 0; ti < dec_tau_samp; ti++) {
        int ti_inv = dec_tau_samp + ti;

        in_tau[ti][0] = in[ti_inv][0] * in[ti][0] + in[ti_inv][1] * in[ti][1];
        in_tau[ti][1] = -in[ti_inv][0] * in[ti][1] + in[ti_inv][1] * in[ti][0];
    }
}

int find_fftwf_peak(fftwf_complex* arr, int len) {
    int index = 0;
    float abs_val;
    float max_val = 0;
    for (int i = 0; i < len; i++) {
        abs_val = arr[i][0] * arr[i][0] + arr[i][1] * arr[i][1];
        if (abs_val > max_val) {
            max_val = abs_val;
            index = i;
        }
    }
    return index;
}

void fft_shift_1d(fftwf_complex* data, int len) {
    // Determine the split point
    int mid = (len + 1) / 2;
    fftwf_complex* tmp = (fftwf_complex*)malloc((size_t)len * sizeof(fftwf_complex));

    memcpy(tmp, data + mid, (size_t)(len - mid) * sizeof(fftwf_complex));
    memcpy(tmp + (len - mid), data, (size_t)mid * sizeof(fftwf_complex));

    memcpy(data, tmp, (size_t)len * sizeof(fftwf_complex));

    free(tmp);
}
