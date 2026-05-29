#include "complex_utils.h"

#include <assert.h>
#include <complex.h>
#include <stdint.h>

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
            correlation += y[j] * conjf(x[i]);
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
    for (int i = min_delay; i < max_delay; i++) {
        crosscorrelate_single_delay(x, size_x, y, size_y, i, &result[i - min_delay]);
    }
}

void set_value_at_indices(float complex* value, int start, int stop, float complex* outarray) {
    for (int i = start; i < stop; i++) {
        outarray[i] = value[0];
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
    for (int i = start; i < stop; i++) {
        outarray[i - start] = inarray[i] * conjf(inarray[i]);
    }
}
