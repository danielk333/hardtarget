LIBS=src/gmf_c_lib src/gmf_cuda_lib

all:
	for dr in $(LIBS); do \
		$(MAKE) -C $$dr all; \
	done

debug:
	for dr in $(LIBS); do \
		$(MAKE) -C $$dr clean; \
		$(MAKE) -C $$dr debug; \
	done

clean:
	for dr in $(LIBS); do \
		$(MAKE) -C $$dr clean; \
	done