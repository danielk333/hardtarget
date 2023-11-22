#include <math.h>

int gmf(float *z_tx, int z_tx_len, float *z_rx, int z_rx_len, float *acc_phasors, int n_accs, int *rgs, int n_rg,
        int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, int *a_vec, int *rx_window);