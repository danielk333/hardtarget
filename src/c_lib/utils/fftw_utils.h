#include <fftw3.h>

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
);
float compute_echo_power(fftwf_complex* echo, int echo_len);
void multiply_acc_phasors(
    fftwf_complex* in,
    fftwf_complex* echo,
    int echo_len,
    float* acc_phasors,
    int phasor_index,
    int* dec_rx_inds
);
void compute_phase_difference(
    fftwf_complex* in,
    int dec_signal_len,
    fftwf_complex* in_tau,
    int dec_tau_samp,
    fftwf_complex* echo,
    int echo_len,
    int* dec_rx_inds
);
int find_fftwf_peak(fftwf_complex* arr, int len);