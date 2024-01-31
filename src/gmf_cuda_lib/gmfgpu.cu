// header file for the plasmaline project
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
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

/*
  For each range gate (i), multiply transmit pulse with range delayed echo
  each range gate has a nfft2 length block in d_z_echo.
  all of these will be fft'ed in parallel with CuFFT at the next stage.

 */
__global__ void form_input(cufftComplex *z_tx, int z_tx_len, int nfft2, cufftComplex *z_rx, int *rgs, int dec,
                           cufftComplex *z_echo, int *rxwin_idx) {
    int rgi = blockIdx.x;
    int rg = rgs[rgi];
    int ind;
    for (int ti = 0; ti < z_tx_len; ti++) {
        ind = rgi * nfft2 + ti / dec;
        z_echo[ind] = cuCaddf(z_echo[ind], cuCmulf(z_tx[ti], z_rx[rg + rxwin_idx[ti]]));
    }
}

/*
  For each range gate (i), find the value of the spectrum that has highest power.
  Also store DC value for noise floor determination.
 */
__global__ void peak_find(cufftComplex *z_out, float *gmf_vec, float *gmf_dc_vec, int *v_vec, int *a_vec, int nfft2,
                          int acc_idx) {
    int rgi = blockIdx.x;
    float this_gmf;
    int ind = rgi * nfft2;
    if (acc_idx == 0) gmf_dc_vec[rgi] = z_out[ind].x * z_out[ind].x + z_out[ind].y * z_out[ind].y;
    for (int j = 0; j < nfft2; j++) {
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
__global__ void phasor_multiply(cufftComplex *z_echo, cufftComplex *z_in, int nfft2, int acc_idx,
                                cufftComplex *acc_phasors) {
    int rgi = blockIdx.x;
    for (int j = 0; j < nfft2; j++) {
        z_in[rgi * nfft2 + j] = cuCmulf(z_echo[rgi * nfft2 + j], acc_phasors[acc_idx * nfft2 + j]);
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
static inline void check_cuda(int res, const char *name, const char *description){
    if (res != cudaSuccess) {
        printf("Error %d\n", res);
        fprintf(stderr, "Cuda error: Failed to %s for variable '%s'\n", description, name);
        exit(EXIT_FAILURE);
    }
}


static inline void check_cudaMalloc(int res, const char *name){
    check_cuda(res, name, "allocate");
}
static inline void check_cudaFree(int res, const char *name){
    check_cuda(res, name, "free");
}
static inline void check_cudaMemcopy(int res, const char *name){
    check_cuda(res, name, "copy values");
}
static inline void check_cudaMemset(int res, const char *name){
    check_cuda(res, name, "set values");
}


/*
   This is the main code. If you have N GPUs, you can run N gmf functions in parallel.

*/
extern "C" int gmf(float *z_tx, int z_tx_len, float *z_rx, int z_rx_len, float *acc_phasors, int n_accs, int *rgs,
                   int n_rg, int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, int *a_vec, int *rx_window,
                   int gpu_id) {
    cudaSetDevice(gpu_id);
    // initializing pointers to device (GPU) memory, denoted with "d_"
    cufftComplex *d_z_tx;
    cufftComplex *d_z_rx;
    cufftComplex *d_z_in;
    cufftComplex *d_z_echo;
    cufftComplex *d_acc_phasors;

    float *d_gmf_vec;
    float *d_gmf_dc_vec;

    int *d_rgs;
    int *d_v_vec;
    int *d_a_vec;
    int *d_rx_window;

    int nfft2 = (int)(z_tx_len / dec);

    // allocating device memory to the above pointers
    // the signal and echo here are only one row of the CPU data (one time step)
    check_cudaMalloc(cudaMalloc((void **)&d_z_tx, sizeof(cufftComplex) * z_tx_len), "d_z_tx");
    check_cudaMalloc(cudaMalloc((void **)&d_z_rx, sizeof(cufftComplex) * z_rx_len), "d_z_rx");
    check_cudaMalloc(cudaMalloc((void **)&d_z_in, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_in");
    check_cudaMalloc(cudaMalloc((void **)&d_z_echo, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_echo");
    check_cudaMalloc(cudaMalloc((void **)&d_acc_phasors, sizeof(cufftComplex) * n_accs * nfft2), "d_acc_phasors");

    check_cudaMalloc(cudaMalloc((void **)&d_gmf_vec, sizeof(float) * n_rg), "d_gmf_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_gmf_dc_vec, sizeof(float) * n_rg), "d_gmf_dc_vec");

    check_cudaMalloc(cudaMalloc((void **)&d_rgs, sizeof(int) * n_rg), "d_rgs");
    check_cudaMalloc(cudaMalloc((void **)&d_v_vec, sizeof(int) * n_rg), "d_v_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_a_vec, sizeof(int) * n_rg), "d_a_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_rx_window, sizeof(int) * z_tx_len), "d_rx_window");

    // initializing 1D FFT plan, this will tell cufft execution how to operate
    // cufft is well optimized and will run with different parameters than above
    cufftHandle plan;
    check_cufft(cufftPlan1d(&plan, nfft2, CUFFT_C2C, n_rg), "Create 1D FFT plan");

    // execution of the prepared kernels n_ipp times
    // ensure empty device spectrum
    check_cudaMemset(cudaMemset(d_z_in, 0, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_in");
    check_cudaMemset(cudaMemset(d_z_echo, 0, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_echo");

    // copying n_ipp'th row of host data into device
    check_cudaMemcopy(cudaMemcpy(d_z_tx, z_tx, sizeof(cufftComplex) * z_tx_len, cudaMemcpyHostToDevice), "d_z_tx");
    check_cudaMemcopy(cudaMemcpy(d_rgs, rgs, sizeof(int) * n_rg, cudaMemcpyHostToDevice), "d_rgs");
    check_cudaMemcopy(cudaMemcpy(d_rx_window, rx_window, sizeof(int) * z_tx_len, cudaMemcpyHostToDevice), "d_rx_window");
    check_cudaMemcopy(cudaMemcpy(d_acc_phasors, acc_phasors, sizeof(cufftComplex) * nfft2*n_accs, cudaMemcpyHostToDevice), "d_acc_phasors");
    check_cudaMemcopy(cudaMemcpy(d_z_rx, z_rx, sizeof(cufftComplex) * z_rx_len, cudaMemcpyHostToDevice), "d_z_rx");

    // form input
    form_input<<<n_rg, 1>>>(d_z_tx, z_tx_len, nfft2, d_z_rx, d_rgs, dec, d_z_echo, d_rx_window);

    for (int acc_idx = 0; acc_idx < n_accs; acc_idx++) {
        check_cudaMemset(cudaMemset(d_z_in, 0, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_in");
        phasor_multiply<<<n_rg, 1>>>(d_z_echo, d_z_in, nfft2, acc_idx, d_acc_phasors);

        // cufft kernel execution
        check_cufft(cufftExecC2C(plan, (cufftComplex *)d_z_in, (cufftComplex *)d_z_in, CUFFT_FORWARD), "ExecC2C Forward");
        peak_find<<<n_rg, 1>>>(d_z_in, d_gmf_vec, d_gmf_dc_vec, d_v_vec, d_a_vec, nfft2, acc_idx);
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
    
    check_cufft(cufftDestroy(plan), "destroy plan");
    return 0;
}