NVCC=nvcc
CFLAGS="-arch=sm_30"
LDFLAGS=-lcublas


demo:
	$(NVCC) -g $(CFLAGS) -o demo wt.cu common.cu utils.cu separable.cu nonseparable.cu haar.cu filters.cpp demo.cpp io.cpp -lcublas


libpdwt.so:
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC' -o $@ --shared wt.cu common.cu utils.cu separable.cu nonseparable.cu haar.cu filters.cpp $(CFLAGS) $(LDFLAGS)


%.o: %.cu
	$(NVCC) -c $(CFLAGS) $<

clean:
	rm -f demo *.o *.so
