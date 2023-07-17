
CC := g++
CFLAGS := 
LDFLAGS := 
LDLIBS := -L. -lATC3DG64 -lws2_32 -liconv

all: trackStarCollector

clean:
	rm --force *.o *.exe

.PHONY: all clean

trakstarcollector: trackStarCollector.o
	$(CC)  -o trackStarCollector show_data_points.o $(LDFLAGS) $(LDLIBS)
	rm show_data_points.o

%.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@



