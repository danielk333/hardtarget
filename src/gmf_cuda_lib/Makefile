CC = nvcc
all: CFLAGS = --compiler-options '-fPIC' -shared
debug: CFLAGS = --compiler-options '-fPIC' -shared -g -G
LIBS = -lcufft
INSTALL = ../hardtarget/gmf

SOURCES=gmfgpu.cu
EXT=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTLIB=gmfcudalib$(EXT)
INSTALLLIB=$(INSTALL)/$(OUTLIB)

all: $(INSTALLLIB) $(OUTLIB)
debug: $(INSTALLLIB) $(OUTLIB)

hello_world: hello.cu
	$(CC) $(CFLAGS) hello.cu -o hello_world

$(OUTLIB): $(SOURCES)
	@echo "gmfcudalib installation -> Linking shared library $(OUTLIB) ..."
	$(CC) $(CFLAGS) $(LIBS) $(SOURCES) -o $(OUTLIB)
	@echo "gmfcudalib installation -> The shared library $(OUTLIB) has been created successfully."

$(INSTALLLIB): $(OUTLIB)
	@cp $(OUTLIB) $(INSTALLLIB) -v
	@echo "gmfcudalib installation -> done"

clean:
	@echo "gmfcudalib clean -> Removing shared library ..."
	@-rm -fv $(OUTLIB)
	@-rm -fv $(INSTALLLIB)