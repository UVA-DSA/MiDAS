
CC := g++
CFLAGS := 
LDFLAGS := 
LDLIBS := -L. -lATC3DG64

all: trackStarCollector

clean:
	rm --force primes.o main.o primes

.PHONY: all clean

trakstarcollector: trackStarCollector.o
	$(CC)  -o trackStarCollector show_data_points.o $(LDFLAGS) $(LDLIBS)

%.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@



