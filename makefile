CXX = nvcc

CXXFLAGS  = -O2 -std=c++11 -I ./eigen -arch sm_30
CSRCS     = $(wildcard *.cpp) $(wildcard *.cu)
CHDRS     = $(wildcard *.h)
COBJS     = $(addsuffix .o, $(basename $(CSRCS)))

all: $(COBJS)
	$(CXX) $(CXXFLAGS) $(COBJS) -o main

FEM.o: FEM.cu FEM.h
	$(CXX) $(CXXFLAGS) FEM.cu -c

run:
	./main

clean:
	rm -rf *.o
	rm main