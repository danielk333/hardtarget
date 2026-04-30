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
    int counter = 0;
    for (int i = max_delay; i > min_delay; i--) {
        crosscorrelate_single_delay(x, size_x, y, size_y, i, &result[counter]);
        counter++;
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
    int counter = 0;
    for (int i = start; i < stop; i++) {
        outarray[counter] = inarray[i] * conj(inarray[i]);
        counter++;
    }
}
