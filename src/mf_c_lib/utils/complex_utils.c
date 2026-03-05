#include "complex_utils.h"
#include <assert.h>



void crosscorrelate_single_delay(
    float complex* x,
    int size_x,
    float complex* y,
    int size_y,
    int delay,
    float complex* result
) {
    float complex correlation = 0.0 + 0.0 * I;

    for (int i = 0; i < size_x; i++) {
        int j = i + delay;
        if (j < 0 || j >= size_y) {
            correlation += 0.0 + 0.0 * I;
        } else {
            correlation += x[i] * conj(y[j]);
        }
    }
    *result = correlation;
}

void crosscorrelate(
    float complex* x,
    int size_x,
    float complex* y,
    int size_y,
    int min_delay,
    int max_delay,
    float complex* result
) {
    assert(max_delay > min_delay);
    int count = 0;
    for (int i = max_delay; i > min_delay; i--) {
        crosscorrelate_single_delay(x, size_x, y, size_y, i, &result[count]);
        count++;
    }
}

// Sets the value of abs_rx_sum from start to stop on outarray
void set_norm_coefs(float complex* abs_rx_sum, int start, int stop, float complex* outarray) {
    for (int i = start; i < stop; i++) {
        outarray[i] = abs_rx_sum[0];
    }
}

float complex complex_sum(float complex* inarray, int size) {
    float complex rv = 0.0 + 0.0 * I;
    for (int i = 0; i < size; i++) {
        rv += inarray[i];
    }
    return rv;
}

void elementwise_cabs_square(float complex* inarray, int start, int stop, float complex* outarray) {
    int temp = 0;
    for (int i = start; i < stop; i++) {
        outarray[temp] = inarray[i] * conj(inarray[i]);
        temp++;
    }
}