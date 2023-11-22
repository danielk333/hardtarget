CC = gcc
CFLAGS = -fPIC -shared -g
LIBS = -lfftw3f
INSTALL = ../hardtarget/gmf

SOURCES=gmf.c
EXT=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTLIB=gmfclib$(EXT)
INSTALLLIB=$(INSTALL)/$(OUTLIB)

all: $(INSTALLLIB) $(OUTLIB)

$(OUTLIB): $(SOURCES)
	@echo "gmfclib installation -> Linking shared library $(OUTLIB) ..."
	$(CC) $(CFLAGS) $(LIBS) $(SOURCES) -o $(OUTLIB)
	@echo "gmfclib installation -> The shared library $(OUTLIB) has been created successfully."

$(INSTALLLIB): $(OUTLIB)
	@cp $(OUTLIB) $(INSTALLLIB) -v
	@echo "gmfclib installation -> done"

clean:
	@echo "gmfclib clean -> Removing shared library ..."
	@-rm -fv $(OUTLIB)
	@-rm -fv $(INSTALLLIB)