CXX = g++

CXXFLAGS  = -O2 -std=c++11 -I ./eigen
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