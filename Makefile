NVCC=nvcc
CFLAGS="-arch=sm_30"
LDFLAGS=-lcublas


demo: wt.o common.o io.o demo.o
	$(NVCC) -g -o $@ $^  $(CFLAGS) $(LDFLAGS)


profiling: wt.o common.o io.o profiling.o
	$(NVCC) -g -o $@ $^  $(CFLAGS) $(LDFLAGS)


libppdwt.so: wt.cu common.cu
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC' -o $@ --shared wt.cu common.cu $(CFLAGS) $(LDFLAGS)


%.o: %.cu
	$(NVCC) -c $(CFLAGS) $<

clean:
	rm -f *.o
