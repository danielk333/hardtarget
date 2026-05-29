#include <assert.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mf.h"
#include "utils/complex_utils.h"

// takes a start and an end and creates an array with
// steps size between them and stores it in outarray
void arange(double start, double end, double step, double* outarray) {
    int counter = 0;
    for (double i = start; i <= end; i += step) {
        outarray[counter] = i;
        counter++;
    }
}

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

) {
    int decoded_size = rx_len + tx_len;
    float complex rx_norm_coefs[decoded_size];
    float complex abs_rx[tx_len];
    float complex output_power[decoded_size];
    float doppler_freq_samp;
    float complex signal_model[tx_len];
    float complex decoded[decoded_size];
    float complex signal_model_abs_arr[tx_len];
    float complex abs_rx_sum;

    // Setting arrays to zeroes
    memset(pows, (float complex)0, (size_t)(pows_size[0] * pows_size[1]));
    memset(pows_normalized, (float complex)0, (size_t)(pows_size[0] * pows_size[1]));
    memset(max_pow_per_doppler, (float complex)0, (size_t)max_pow_per_doppler_size);
    memset(max_pow_ind, (int)0, (size_t)max_pow_ind_size);

    // Calculate the normalization coefficient with a sliding window of size tx_len
    for (int i = tx_len; i <= rx_len; i++) {
        elementwise_cabs_square(rx, i - tx_len, i, abs_rx);
        abs_rx_sum = complex_sum(abs_rx, tx_len);
        if (cabsf(abs_rx_sum) >= FLT_EPSILON) {
            rx_norm_coefs[i] = sqrtf(abs_rx_sum);
        } else {
            rx_norm_coefs[i] = 1;
        }
    }

    // Set first tx_len datapoints to the norm coefs of the first window
    set_value_at_indices(&rx_norm_coefs[tx_len], 0, tx_len, rx_norm_coefs);
    // Set the last tx_len datapoints to the norm coefs of the last valid window
    set_value_at_indices(&rx_norm_coefs[rx_len], rx_len, rx_len + tx_len, rx_norm_coefs);

    // Calculate correlation for each doppler frequency
    for (int i = 0; i < doppler_frequencies_len; i++) {
        // Calculate tx signal model
        for (int j = 0; j < tx_len; j++) {
            doppler_freq_samp = 2 * M_PI * doppler_frequencies[i] * (j * range_gate_step + 1) * t_samp_usec * 1e-6;
            signal_model[j] = tx[j] * (cosf(doppler_freq_samp) + I * sinf(doppler_freq_samp));
        }

        // Cross correlate rx and tx signal
        crosscorrelate(signal_model, tx_len, rx, rx_len, -tx_len, rx_len, decoded);

        // Calculate tx normalization coefficient
        elementwise_cabs_square(signal_model, 0, tx_len, signal_model_abs_arr);
        float complex tx_norm_coefs = sqrtf(complex_sum(signal_model_abs_arr, tx_len));

        // Find index with best match from cross correlation
        for (int j = 0; j < decoded_size; j++) {
            output_power[j] = cpowf(cabsf(decoded[j] / (rx_norm_coefs[j] * tx_norm_coefs)), 2);

            if (cabsf(output_power[j]) > cabsf(max_pow_per_doppler[i])) {
                max_pow_per_doppler[i] = output_power[j];
                max_pow_ind[i] = j;
            }

            pows_normalized[i * decoded_size + j] = output_power[j];
            pows[i * decoded_size + j] = decoded[j];
        }
    }
    return 0;
}
