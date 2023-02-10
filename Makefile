
CC := clang
CFLAGS := 
LDFLAGS := -Wall
LDLIBS := 

all: trakstarcollector

clean:
	rm --force primes.o main.o primes

.PHONY: all clean

%.o: %.c %.h primes.h
	$(CC) $(CFLAGS) -c $< -o $@
