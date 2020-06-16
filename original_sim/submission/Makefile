GCC_92_PATH=/local/java/gcc-9.2.0/bin/gcc
ifneq ("$(wildcard $(GCC_92_PATH))", "")
	CC=$(GCC_92_PATH)
else
	CC=gcc
endif

CFLAGS_VECTOR=-msse3 -mssse3 -msse2 -msse -msse4.1 -msse4.2 -mavx -mavx2 -mfma
CFLAGS=-O2 -Wall -g -fopenmp -Wno-unused-result $(CFLAGS_VECTOR) -ffp-contract=off

.c.o:
	$(CC) -c $(CFLAGS) $<

all: cfd bin2ppm

clean:
	rm -f bin2ppm cfd *.o output.bin output.ppm

cfd: utilities.o simulation.o constants.h cfd.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

bin2ppm: bin2ppm.o utilities.o constants.h
	$(CC) $(CFLAGS) -o $@ $^ -lm
