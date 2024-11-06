#include <cstdio>
#include <cstring>
#include "gmfgpu.h"

extern "C" void print_devices() {
    int nDevices;

    int ret = cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("\n");
    }
}

/*
  For each range gate (i), multiply transmit pulse with range delayed echo
  each range gate has a echo_len length block in d_z_echo.
  all of these will be fft'ed in parallel in an isochronus array with CuFFT at the next stage.

 */
__global__ void form_input(cufftComplex *z_tx, int z_tx_len, int echo_len, cufftComplex *z_rx, int *rgs, int dec,
                           cufftComplex *z_echo, int *rxwin_idx) {
    int rgi = blockIdx.x;
    int rg = rgs[rgi];
    int ind;
    for (int ti = 0; ti < z_tx_len; ti++) {
        ind = rgi * echo_len + ti / dec;
        z_echo[ind] = cuCaddf(z_echo[ind], cuCmulf(z_tx[ti], z_rx[rg + rxwin_idx[ti]]));
    }
}

/*
  For each range gate (i), find the value of the spectrum that has highest power.
  Also store DC value for noise floor determination.
 */
__global__ void peak_find(cufftComplex *z_out, float *gmf_vec, float *gmf_dc_vec, int *v_vec, int *a_vec, int n_fft,
                          int acc_idx) {
    int rgi = blockIdx.x;
    float this_gmf;
    int ind = rgi * n_fft;
    // zero-frequency (DC) component in cufft out[0] according to docs
    if (acc_idx == 0) gmf_dc_vec[rgi] = z_out[ind].x * z_out[ind].x + z_out[ind].y * z_out[ind].y;

    for (int j = 0; j < n_fft; j++) {
        this_gmf = z_out[ind + j].x * z_out[ind + j].x + z_out[ind + j].y * z_out[ind + j].y;
        if (this_gmf > gmf_vec[rgi]) {
            gmf_vec[rgi] = this_gmf;
            v_vec[rgi] = j;
            a_vec[rgi] = acc_idx;
        }
    }
}

/*
  Multiply echo with acceleration phasor, store input in d_z_in
 */
__global__ void phasor_multiply(cufftComplex *z_echo, cufftComplex *z_in, int echo_len, int dec_signal_len, int acc_idx,
                                cufftComplex *acc_phasors, int *d_dec_rx_inds) {
    int rgi = blockIdx.x;
    int target_index;
    for (int j = 0; j < echo_len; j++) {
        target_index = rgi * dec_signal_len + d_dec_rx_inds[j];
        z_in[target_index] = cuCmulf(z_echo[rgi * echo_len + j], acc_phasors[acc_idx * echo_len + j]);
    }
}

/*
  Check that cuda function did not fail with custom message
 */
static inline void check_cufft(int res, const char *description){
    if (res != CUFFT_SUCCESS) {
        printf("Error %d\n", res);
        fprintf(stderr, "Cuda FFT error: Failed to %s\n", description);
        exit(EXIT_FAILURE);
    }
}

/*
  Check that cuda function did not fail with custom message
 */
static inline void check_cuda(cudaError_t res, const char *name, const char *description){
    if (res != cudaSuccess) {
        printf("Error %d = '%s'\n", res, cudaGetErrorString(res));
        fprintf(stderr, "Cuda error: Failed to %s for variable '%s'\n", description, name);
        exit(EXIT_FAILURE);
    }
}


static inline void check_cudaMalloc(cudaError_t res, const char *name){
    check_cuda(res, name, "allocate");
}
static inline void check_cudaFree(cudaError_t res, const char *name){
    check_cuda(res, name, "free");
}
static inline void check_cudaMemcopy(cudaError_t res, const char *name){
    check_cuda(res, name, "copy values");
}
static inline void check_cudaMemset(cudaError_t res, const char *name){
    check_cuda(res, name, "set values");
}

/*
   This is the main code. If you have N GPUs, you can run N gmf functions in parallel.

    The commented numbers are the argument numbers, useful for debugging the ctypes interface.
*/
extern "C" int gmf(
    float *z_tx, int z_tx_len, // 1, 2
    float *z_rx, int z_rx_len, // 3, 4
    float *acc_phasors, int n_accs, // 5, 6
    int *rgs, int n_rg, // 7, 8
    int dec, // 9
    float *gmf_vec, // 10
    float *gmf_dc_vec, // 11
    int *v_vec, // 12
    int *a_vec, // 13
    int *rx_window, // 14
    int *dec_rx_inds, // 15
    int dec_signal_len, // 16
    int gpu_id // 17
) {
    cudaSetDevice(gpu_id);
    // initializing pointers to device (GPU) memory, denoted with "d_"
    cufftComplex *d_z_tx;
    cufftComplex *d_z_rx;
    cufftComplex *d_z_in;
    cufftComplex *d_z_echo;
    cufftComplex *d_acc_phasors;

    int echo_len = (int)(z_tx_len / dec);

    float *d_gmf_vec;
    float *d_gmf_dc_vec;

    int *d_rgs;
    int *d_v_vec;
    int *d_a_vec;
    int *d_rx_window;
    int *d_dec_rx_inds;


    // allocating device memory to the above pointers
    // the signal and echo here are only one row of the CPU data (one time step)
    check_cudaMalloc(cudaMalloc((void **)&d_z_tx, sizeof(cufftComplex) * z_tx_len), "d_z_tx");
    check_cudaMalloc(cudaMalloc((void **)&d_z_rx, sizeof(cufftComplex) * z_rx_len), "d_z_rx");
    check_cudaMalloc(cudaMalloc((void **)&d_z_in, sizeof(cufftComplex) * dec_signal_len * n_rg), "d_z_in");
    check_cudaMalloc(cudaMalloc((void **)&d_z_echo, sizeof(cufftComplex) * echo_len * n_rg), "d_z_echo");
    check_cudaMalloc(cudaMalloc((void **)&d_acc_phasors, sizeof(cufftComplex) * n_accs * echo_len), "d_acc_phasors");

    check_cudaMalloc(cudaMalloc((void **)&d_gmf_vec, sizeof(float) * n_rg), "d_gmf_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_gmf_dc_vec, sizeof(float) * n_rg), "d_gmf_dc_vec");

    check_cudaMalloc(cudaMalloc((void **)&d_rgs, sizeof(int) * n_rg), "d_rgs");
    check_cudaMalloc(cudaMalloc((void **)&d_v_vec, sizeof(int) * n_rg), "d_v_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_a_vec, sizeof(int) * n_rg), "d_a_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_rx_window, sizeof(int) * z_tx_len), "d_rx_window");
    check_cudaMalloc(cudaMalloc((void **)&d_dec_rx_inds, sizeof(int) * echo_len), "d_dec_rx_inds");


    // initializing 1D FFT plan, this will tell cufft execution how to operate
    // cufft is well optimized and will run with different parameters than above
    cufftHandle plan;
    check_cufft(cufftPlan1d(&plan, dec_signal_len, CUFFT_C2C, n_rg), "Create 1D FFT plan");

    // execution of the prepared kernels n_ipp times
    // ensure empty device spectrum
    check_cudaMemset(cudaMemset(d_z_in, 0, sizeof(cufftComplex) * dec_signal_len * n_rg), "d_z_in");
    check_cudaMemset(cudaMemset(d_z_echo, 0, sizeof(cufftComplex) * echo_len * n_rg), "d_z_echo");

    // copying n_ipp'th row of host data into device
    check_cudaMemcopy(cudaMemcpy(d_z_tx, z_tx, sizeof(cufftComplex) * z_tx_len, cudaMemcpyHostToDevice), "d_z_tx");
    check_cudaMemcopy(cudaMemcpy(d_rgs, rgs, sizeof(int) * n_rg, cudaMemcpyHostToDevice), "d_rgs");
    check_cudaMemcopy(cudaMemcpy(d_rx_window, rx_window, sizeof(int) * z_tx_len, cudaMemcpyHostToDevice), "d_rx_window");
    check_cudaMemcopy(cudaMemcpy(d_dec_rx_inds, dec_rx_inds, sizeof(int) * echo_len, cudaMemcpyHostToDevice), "d_dec_rx_inds");
    check_cudaMemcopy(cudaMemcpy(d_acc_phasors, acc_phasors, sizeof(cufftComplex) * echo_len * n_accs, cudaMemcpyHostToDevice), "d_acc_phasors");
    check_cudaMemcopy(cudaMemcpy(d_z_rx, z_rx, sizeof(cufftComplex) * z_rx_len, cudaMemcpyHostToDevice), "d_z_rx");

    // form input
    form_input<<<n_rg, 1>>>(d_z_tx, z_tx_len, echo_len, d_z_rx, d_rgs, dec, d_z_echo, d_rx_window);

    for (int acc_idx = 0; acc_idx < n_accs; acc_idx++) {
        check_cudaMemset(cudaMemset(d_z_in, 0, sizeof(cufftComplex) * dec_signal_len * n_rg), "d_z_in");
        phasor_multiply<<<n_rg, 1>>>(d_z_echo, d_z_in, echo_len, dec_signal_len, acc_idx, d_acc_phasors, d_dec_rx_inds);

        // cufft kernel execution
        check_cufft(cufftExecC2C(plan, (cufftComplex *)d_z_in, (cufftComplex *)d_z_in, CUFFT_FORWARD), "ExecC2C Forward");
        peak_find<<<n_rg, 1>>>(d_z_in, d_gmf_vec, d_gmf_dc_vec, d_v_vec, d_a_vec, dec_signal_len, acc_idx);
    }

    // copying device resultant spectrum to host, now able to be manipulated
    check_cudaMemcopy(cudaMemcpy(gmf_vec, d_gmf_vec, sizeof(float) * n_rg, cudaMemcpyDeviceToHost), "gmf_vec");
    check_cudaMemcopy(cudaMemcpy(gmf_dc_vec, d_gmf_dc_vec, sizeof(float) * n_rg, cudaMemcpyDeviceToHost), "gmf_dc_vec");
    check_cudaMemcopy(cudaMemcpy(v_vec, d_v_vec, sizeof(int) * n_rg, cudaMemcpyDeviceToHost), "v_vec");
    check_cudaMemcopy(cudaMemcpy(a_vec, d_a_vec, sizeof(int) * n_rg, cudaMemcpyDeviceToHost), "a_vec");

    // memory clean up
    check_cudaFree(cudaFree(d_z_tx), "d_z_tx");
    check_cudaFree(cudaFree(d_rgs), "d_rgs");
    check_cudaFree(cudaFree(d_z_rx), "d_z_rx");
    check_cudaFree(cudaFree(d_z_in), "d_z_in");
    check_cudaFree(cudaFree(d_z_echo), "d_z_echo");
    check_cudaFree(cudaFree(d_v_vec), "d_v_vec");
    check_cudaFree(cudaFree(d_a_vec), "d_a_vec");
    check_cudaFree(cudaFree(d_gmf_vec), "d_gmf_vec");
    check_cudaFree(cudaFree(d_gmf_dc_vec), "d_gmf_dc_vec");
    check_cudaFree(cudaFree(d_acc_phasors), "d_acc_phasors");
    check_cudaFree(cudaFree(d_rx_window), "d_rx_window");
    check_cudaFree(cudaFree(d_dec_rx_inds), "d_dec_rx_inds");

    check_cufft(cufftDestroy(plan), "destroy plan");
    return EXIT_SUCCESS;
}