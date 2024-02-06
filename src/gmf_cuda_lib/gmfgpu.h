#ifndef PLASMA_LINE
#define PLASMA_LINE

// system includes
#include <stdio.h>
#include <time.h>

// CUDA includes
#include <cuComplex.h>
#include <cufft.h>

extern "C" void print_devices();
extern "C" int gmf(
    float *z_tx, int z_tx_len, float *z_rx, int z_rx_len,
    float *acc_phasors, int n_accs, int *rgs, int n_rg,
    int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, 
    int *a_vec, int *rx_window, int *dec_rx_inds, int dec_signal_len,
    int gpu_id
);

#endif
