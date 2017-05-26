CXX = g++

CXXFLAGS  = -Wall -O2 -std=c++11 -Ieigen/
CSRCS     = $(wildcard *.cpp)
CHDRS     = $(wildcard *.h)
COBJS     = $(addsuffix .o, $(basename $(CSRCS)))

all: $(COBJS)
	$(CXX) $(CFLAGS) $(COBJS) -o main

run:
	./main

clean:
	rm -rf *.o
	rm main