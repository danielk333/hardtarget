CC = gcc
all: CFLAGS = -fPIC -shared
debug: CFLAGS = -fPIC -shared -g
LIBS = -lfftw3f
INSTALL = ../hardtarget/libs

SOURCES=gmf.c, dpt.c, xcorr.c
EXT=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTLIB=mfclib$(EXT)
INSTALLLIB=$(INSTALL)/$(OUTLIB)

all: $(INSTALLLIB) $(OUTLIB)
debug: $(INSTALLLIB) $(OUTLIB)

$(OUTLIB): $(SOURCES)
	@echo "clib installation -> Linking shared library $(OUTLIB) ..."
	$(CC) $(CFLAGS) $(LIBS) $(SOURCES) -o $(OUTLIB)
	@echo "clib installation -> The shared library $(OUTLIB) has been created successfully."

$(INSTALLLIB): $(OUTLIB)
	@cp $(OUTLIB) $(INSTALLLIB) -v
	@echo "clib installation -> done"

clean:
	@echo "clib clean -> Removing shared library ..."
	@-rm -fv $(OUTLIB)
	@-rm -fv $(INSTALLLIB)