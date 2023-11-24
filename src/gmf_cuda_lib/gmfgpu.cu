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
                           cufftComplex *d_z_echo, int *d_rxwin_idx) {
    int i = blockIdx.x;
    int rg = rgs[i];
    for (int ti = 0; ti < z_tx_len; ti++) {
        d_z_echo[i * nfft2 + ti / dec] = cuCaddf(
            d_z_echo[i * nfft2 + ti / dec], 
            cuCmulf(z_tx[ti], z_rx[rg + d_rxwin_idx[ti]])
        );
    }
}

/*
  For each range gate (i), find the value of the spectrum that has highest power.
  Also store DC value for noise floor determination.
 */
__global__ void peak_find(cufftComplex *z_out, float *gmf_vec, float *gmf_dc_vec, int *v_vec, int *a_vec, int nfft2,
                          int acc_idx) {
    int i = blockIdx.x;
    if (acc_idx == 0) gmf_dc_vec[i] = z_out[i * nfft2].x * z_out[i * nfft2].x + z_out[i * nfft2].y * z_out[i * nfft2].y;
    for (int j = 0; j < nfft2; j++) {
        float this_gmf =
            z_out[i * nfft2 + j].x * z_out[i * nfft2 + j].x + z_out[i * nfft2 + j].y * z_out[i * nfft2 + j].y;
        if (this_gmf > gmf_vec[i]) {
            gmf_vec[i] = this_gmf;
            v_vec[i] = j;
            a_vec[i] = acc_idx;
        }
    }
}

/*
  Multiply echo with acceleration phasor, store input in d_z_in
 */
__global__ void phasor_multiply(cufftComplex *d_z_echo, cufftComplex *d_z_in, int nfft2, int i,
                                cufftComplex *d_acc_phasors) {
    int rgi = blockIdx.x;
    /*
       TODO: only multiply non-zero values.
    */
    for (int j = 0; j < nfft2; j++) {
        d_z_in[rgi * nfft2 + j] = cuCmulf(d_z_echo[rgi * nfft2 + j], d_acc_phasors[i * nfft2 + j]);
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

    TODO: fix docstring in code and print statements to print better messages
    TODO: refactor out deallocs and allocs into a function call

*/
extern "C" int gmf(float *z_tx, int z_tx_len, float *z_rx, int z_rx_len, float *acc_phasors, int n_accs, int *rgs,
                   int n_rg, int dec, float *gmf_vec, float *gmf_dc_vec, int *v_vec, int *a_vec, int *rx_window,
                   int gpu_id) {
    cudaSetDevice(gpu_id);
    // initializing pointers to device (GPU) memory, denoted with "d_"
    cufftComplex *d_z_tx;
    cufftComplex *d_z_rx;
    cufftComplex *d_z_echo;
    cufftComplex *d_z_in;
    cufftComplex *d_acc_phasors;
    float *d_gmf_vec;
    float *d_gmf_dc_vec;
    int *d_v_vec;
    int *d_a_vec;
    int *d_rx_window;

    int *d_rgs;
    int nfft2;

    nfft2 = (int)(z_tx_len / dec);

    // allocating device memory to the above pointers
    // the signal and echo here are only one row of the CPU data (one time step)
    check_cudaMalloc(cudaMalloc((void **)&d_z_tx, sizeof(cufftComplex) * z_tx_len), "d_z_tx");
    check_cudaMalloc(cudaMalloc((void **)&d_rgs, sizeof(int) * n_rg), "d_rgs");
    check_cudaMalloc(cudaMalloc((void **)&d_rx_window, sizeof(int) * z_tx_len), "d_rx_window");
    check_cudaMalloc(cudaMalloc((void **)&d_z_rx, sizeof(cufftComplex) * z_rx_len), "d_z_rx");
    check_cudaMalloc(cudaMalloc((void **)&d_z_in, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_in");
    check_cudaMalloc(cudaMalloc((void **)&d_z_echo, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_echo");
    check_cudaMalloc(cudaMalloc((void **)&d_acc_phasors, sizeof(cufftComplex) * n_accs * nfft2), "d_acc_phasors");
    check_cudaMalloc(cudaMalloc((void **)&d_gmf_vec, sizeof(float) * n_rg), "d_gmf_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_gmf_dc_vec, sizeof(float) * n_rg), "d_gmf_dc_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_v_vec, sizeof(int) * n_rg), "d_v_vec");
    check_cudaMalloc(cudaMalloc((void **)&d_a_vec, sizeof(int) * n_rg), "d_a_vec");

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

    for (int i = 0; i < n_accs; i++) {
        check_cudaMemset(cudaMemset(d_z_in, 0, sizeof(cufftComplex) * nfft2 * n_rg), "d_z_in");
        phasor_multiply<<<n_rg, 1>>>(d_z_echo, d_z_in, nfft2, i, d_acc_phasors);

        // cufft kernel execution
        check_cufft(cufftExecC2C(plan, (cufftComplex *)d_z_in, (cufftComplex *)d_z_in, CUFFT_FORWARD), "ExecC2C Forward");
        peak_find<<<n_rg, 1>>>(d_z_in, d_gmf_vec, d_gmf_dc_vec, d_v_vec, d_a_vec, nfft2, i);
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