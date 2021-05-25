GPU_CC = nvcc
CC=gcc
CFLAGS =
GPU_SO_CFLAGS = --compiler-options '-fPIC' -shared -L /usr/local/cuda-10.1/targets/x86_64-linux/lib/
GPU_LDFLAGS =  -lcufft
SO_CFLAGS = -fPIC -shared
LDFLAGS = -lfftw3f

ALL: libgmf.so

#libgmfgpu.so: gmfgpu.cu
#	$(GPU_CC) $(GPU_SO_CFLAGS) $(GPU_LDFLAGS) gmfgpu.cu -o $@
libgmf.so: gmf.c
	$(CC) $(SO_CFLAGS)  gmf.c -o $@ $(LDFLAGS)

libgmfgpu.so: gmfgpu.cu
	$(GPU_CC) $(GPU_SO_CFLAGS)  gmfgpu.cu -o $@ $(GPU_LDFLAGS)
