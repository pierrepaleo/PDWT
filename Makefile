NVCC=nvcc
CFLAGS="-arch=sm_30"
LDFLAGS=-lcublas


wt: wt.o
	$(NVCC) -g -o $@ $^  $(CFLAGS) $(LDFLAGS)

%.o: %.cu
	$(NVCC) -c $(CFLAGS) $<

clean:
	rm -f *.o
