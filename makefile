CXX = nvcc

CXXFLAGS  = -O2 -std=c++11 -I ./eigen -arch sm_61 --expt-relaxed-constexpr 
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
	rm -rf out*
	rm -rf mesh* 
	rm main

cleandata:
	rm -rf out*
	rm -rf mesh* 
