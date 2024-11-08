#include <math.h>
#include <fftw3.h>

void compute_echo_signal(
    fftwf_complex *echo, int echo_len,
    float *z_tx, int z_tx_len,
    float *z_rx, int z_rx_len,
    int decimation,
    int rg,
    int *rx_window
);
float compute_echo_power(
    fftwf_complex *echo, int echo_len
);
void multiply_acc_phasors(
    fftwf_complex *in,
    fftwf_complex *echo, int echo_len,
    float *acc_phasors,
    int phasor_index,
    int *dec_rx_inds
);
void compute_phase_difference(
    fftwf_complex *in, int dec_signal_len,
    fftwf_complex *in_tau, int dec_tau_samp,
    fftwf_complex *echo, int echo_len,
    int *dec_rx_inds
);
int find_fftwf_peak(fftwf_complex *arr, int len);

int gmf(
    float *z_tx, int z_tx_len, float *z_rx, int z_rx_len,
    float *acc_phasors, int n_accs, int *rgs, int n_rg,
    int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, 
    int *a_vec, int *rx_window, int *dec_rx_inds, int dec_signal_len
);
int dpt(
    float *z_tx, int z_tx_len, float *z_rx, int z_rx_len,
    float *acc_phasors, int n_accs, int *rgs, int n_rg,
    int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, 
    int *a_vec, int *rx_window, int *dec_rx_inds, int dec_signal_len,
    int dec_tau_samp
);