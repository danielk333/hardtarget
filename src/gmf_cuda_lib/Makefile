NVCC=nvcc

HOST_COMPILER=${NVCC_CCBIN}
GPU_COMPUTE_VER=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
GPU_COMPUTE=$(subst .,,$(GPU_COMPUTE_VER))
ARCH_FLAGS=-gencode arch=compute_${GPU_COMPUTE},code=sm_${GPU_COMPUTE}

CCOPTS=--compiler-options '-fPIC'
all: CFLAGS=${CCOPTS} -shared
debug: CFLAGS=${CCOPTS} -shared -g -G
NVFLAGS=-ccbin ${HOST_COMPILER}

LIBS = -lcufft
INSTALL = ../hardtarget/gmf

SOURCES=gmfgpu.cu
EXT=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTLIB=gmfcudalib$(EXT)
INSTALLLIB=$(INSTALL)/$(OUTLIB)

all: $(INSTALLLIB) $(OUTLIB)
debug: $(INSTALLLIB) $(OUTLIB)

hello_world: hello.cu
	$(NVCC) $(CCOPTS) $(NVFLAGS) $(ARCH_FLAGS) hello.cu -o hello_world

$(OUTLIB): $(SOURCES)
	@echo "gmfcudalib installation -> Linking shared library $(OUTLIB) ..."
	$(NVCC) $(CFLAGS) $(NVFLAGS) $(ARCH_FLAGS) $(LIBS) $(SOURCES) -o $(OUTLIB)
	@echo "gmfcudalib installation -> The shared library $(OUTLIB) has been created successfully."

$(INSTALLLIB): $(OUTLIB)
	@cp $(OUTLIB) $(INSTALLLIB) -v
	@echo "gmfcudalib installation -> done"

clean:
	@echo "gmfcudalib clean -> Removing shared library ..."
	@-rm -fv $(OUTLIB)
	@-rm -fv $(INSTALLLIB)