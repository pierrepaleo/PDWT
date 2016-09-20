NVCC=nvcc
CFLAGS="-arch=sm_30"
LDFLAGS=-lcublas

PDWTCORE=src/wt.cu src/common.cu src/utils.cu src/separable.cu src/nonseparable.cu src/haar.cu src/filters.cpp


demo:
	$(NVCC) -g $(CFLAGS) -o build/demo $(PDWTCORE) src/demo.cpp src/io.cpp -lcublas


libpdwt.so:
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC' -o build/$@ --shared $(PDWTCORE) $(CFLAGS) $(LDFLAGS)


%.o: %.cu
	$(NVCC) -c $(CFLAGS) $<

clean:
	rm -f demo *.o *.so
