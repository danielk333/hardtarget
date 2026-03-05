CC = gcc
all: CFLAGS = -fPIC -shared
debug: CFLAGS = -fPIC -shared -g
LIBS = -lfftw3f
INSTALL = ../hardtarget/matched_filter/utils

SOURCES=gmf.c, dpt.c, xcorr.c
EXT=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTLIB=mfclib$(EXT)
INSTALLLIB=$(INSTALL)/$(OUTLIB)

all: $(INSTALLLIB) $(OUTLIB)
debug: $(INSTALLLIB) $(OUTLIB)

$(OUTLIB): $(SOURCES)
	@echo "mfclib installation -> Linking shared library $(OUTLIB) ..."
	$(CC) $(CFLAGS) $(LIBS) $(SOURCES) -o $(OUTLIB)
	@echo "mfclib installation -> The shared library $(OUTLIB) has been created successfully."

$(INSTALLLIB): $(OUTLIB)
	@cp $(OUTLIB) $(INSTALLLIB) -v
	@echo "mfclib installation -> done"

clean:
	@echo "mfclib clean -> Removing shared library ..."
	@-rm -fv $(OUTLIB)
	@-rm -fv $(INSTALLLIB)