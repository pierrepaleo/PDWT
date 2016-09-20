NVCC=nvcc
CFLAGS="-arch=sm_30"
LDFLAGS=-lcublas


#~ demo: wt.o common.o io.o demo.o
#~ 	$(NVCC) -g -o $@ $^  $(CFLAGS) $(LDFLAGS)

demo:
	$(NVCC) -g $(CFLAGS) -o demo wt.cu common.cu utils.cu separable.cu nonseparable.cu haar.cu filters.cpp demo.cpp io.cpp -lcublas


lib:
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC' -o $@ --shared wt.cu common.cu utils.cu separable.cu nonseparable.cu haar.cu filters.cpp $(CFLAGS) $(LDFLAGS)


%.o: %.cu
	$(NVCC) -c $(CFLAGS) $<

clean:
	rm -f *.o *.so
