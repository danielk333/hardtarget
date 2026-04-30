#include <complex.h>



void crosscorrelate_single_delay(
    float complex* x,
    int size_x,
    float complex* y,
    int size_y,
    int delay,
    float complex* result
);

void crosscorrelate(
    float complex* x,
    int size_x,
    float complex* y,
    int size_y,
    int min_delay,
    int max_delay,
    float complex* result
);

void set_value_at_indices(float complex* value, int start, int stop, float complex* outarray);

float complex complex_sum(float complex* inarray, int size);

void elementwise_cabs_square(float complex* inarray, int start, int stop, float complex* outarray);
