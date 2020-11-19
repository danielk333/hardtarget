GPU_CC = nvcc
CC=gcc
CFLAGS =
GPU_SO_CFLAGS = --compiler-options '-fPIC' -shared
GPU_LDFLAGS = -lcufft
SO_CFLAGS = -fPIC -shared
LDFLAGS = -lfftw3f

ALL: libgmfgpu.so libgmf.so

libgmfgpu.so: gmfgpu.cu
	$(GPU_CC) $(GPU_SO_CFLAGS) $(GPU_LDFLAGS) gmfgpu.cu -o $@
libgmf.so: gmf.c
	$(CC) $(SO_CFLAGS) $(LDFLAGS) gmf.c -o $@

