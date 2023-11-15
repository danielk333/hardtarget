LIBS=src/gmf_c_lib src/gmf_cuda_lib

all:
	for dr in $(LIBS); do \
		$(MAKE) -C $$dr; \
	done
