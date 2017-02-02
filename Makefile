NVCC=nvcc
CFLAGS="-arch=sm_30"
LDFLAGS=-lcublas

PDWTCORE=src/wt.cu src/common.cu src/utils.cu src/separable.cu src/nonseparable.cu src/haar.cu src/filters.cpp
PDWTOBJ=build/wt.o build/common.o build/utils.o build/separable.o build/nonseparable.o build/haar.o build/filters.o

#
# Using constant memory accross several files requires to use separate compilation (relocatable device code),
# Otherwise a new constant memory buffer is created for each file (even if the symbol is defined in a common file).
# This was fine until the introduction of Wavelets::set_filters().
# As constant memory is managed through the use of "symbols" rather than buffers, another strategy would be to
# get the pointer address with cudaGetSymbolAddress().
# However, separate compilation might be the way to go for better modularity, easier refactoring and compilation speed.
#
demo: $(PDWTCORE) src/demo.cpp src/io.cpp
	mkdir -p build
	$(NVCC) -g $(CFLAGS) -odir build -dc $^
	$(NVCC) $(CFLAGS) -o demo $(PDWTOBJ) build/demo.o build/io.o $(LDFLAGS)


libpdwt.so: $(PDWTCORE)
	mkdir -p build
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC' -odir build -dc $^
	$(NVCC) $(CFLAGS)  -o $@ --shared $(PDWTOBJ) $(LDFLAGS)


# Double precision library
libpdwtd.so: $(PDWTCORE)
	mkdir -p build
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC' -DDOUBLEPRECISION -odir build -dc $^
	$(NVCC) $(CFLAGS)  -o $@ --shared $(PDWTOBJ) $(LDFLAGS)



clean:
	rm -rf build
