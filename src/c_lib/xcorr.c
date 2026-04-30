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
    for (double i = start; i < (end + step); i += step) {
        outarray[counter] = i;
        counter++;
    }
}

int xcorr_echo_search(
    float complex* tx,                      // 1 input
    int tx_len,                              // 2 size
    float complex* rx,                      // 3 input
    int rx_len,                              // 4 size
    int doppler_freq_min,                    // 5 input
    int doppler_freq_max,                    // 6 input
    int doppler_freq_step,                   // 7 input
    int doppler_freq_size,                   // 8 size
    int t_samp_usec,                         // 9 input
    float complex* pows,                    // 10 output
    int* pows_size,                          // 11 size
    float complex* pows_normalized,         // 12 output
    int* pows_normalized_size,               // 13 size
    float complex* max_pow_per_doppler,     // 14 output
    int max_pow_per_doppler_size,            // 15 size
    int* max_pow_ind,                        // 16 output
    int max_pow_ind_size                     // 17 size


) {
    // Initiate variables
    double doppler_freq[doppler_freq_size];
    arange(doppler_freq_min, doppler_freq_max, doppler_freq_step, doppler_freq);

    int decoded_size = rx_len + tx_len;
    float complex norm_coefs[decoded_size];
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
    memset(norm_coefs, (float complex)0, (size_t)decoded_size);

    // Calculate the absolute value complex sum of rx with a sliding window of size tx_len
    for (int i = tx_len; i <= rx_len; i++) {
        elementwise_cabs_square(rx, i - tx_len, i, abs_rx);
        abs_rx_sum = complex_sum(abs_rx, tx_len);
        norm_coefs[i] = abs_rx_sum;
    }

    // Set first tx_len datapoints to the norm coefs of the first window
    set_value_at_indices(&norm_coefs[tx_len], 0, tx_len, norm_coefs);
    // Set the last tx_len datapoints to the norm coefs of the last valid window
    set_value_at_indices(&norm_coefs[rx_len], rx_len, rx_len + tx_len, norm_coefs);

    // For each doppler frequency
    for (int i = 0; i < doppler_freq_size; i++) {

        // Calculate tx signal model
        for (int j = 0; j < tx_len; j++) {
            doppler_freq_samp = (j + 1) * 2 * M_PI * doppler_freq[i] * t_samp_usec *1e6;
            signal_model[j] = (sin(doppler_freq_samp) * I + cos(doppler_freq_samp)) * tx[j];
        }

        // Calculate tx signal absolute sum
        elementwise_cabs_square(signal_model, 0, tx_len, signal_model_abs_arr);
        double complex signal_model_abs_sum = complex_sum(signal_model_abs_arr, tx_len);

        // Cross correlate rx and tx signal
        crosscorrelate(
            rx, rx_len, signal_model, tx_len, -rx_len, tx_len, decoded
        );

        // Find index with best match from cross correlation
        for (int j = 0; j < decoded_size; j++) {
            if (cabs(norm_coefs[j]) < FLT_EPSILON) {
                norm_coefs[j] = 1;
            }

            output_power[j] = decoded[j] / (sqrt(norm_coefs[j]) * sqrt(signal_model_abs_sum));
            output_power[j] = cpow(cabs(output_power[j]), 2);

            if (cabs(output_power[j]) > cabs(max_pow_per_doppler[i])) {
                max_pow_per_doppler[i] = output_power[j];
                max_pow_ind[i] = j;
            }

            pows_normalized[i * decoded_size + j] = output_power[j];
            pows[i * decoded_size + j] = decoded[j];
        }

        max_pow_ind[i] = max_pow_ind[i] - tx_len;
    }
    return 0;
}
