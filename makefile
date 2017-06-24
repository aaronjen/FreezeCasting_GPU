CXX = nvcc

CXXFLAGS  = -O2 -std=c++11 -I ./eigen -arch sm_30
CSRCS     = $(wildcard *.cpp) $(wildcard *.cu)
CHDRS     = $(wildcard *.h)
COBJS     = $(addsuffix .o, $(basename $(CSRCS)))

all: $(COBJS)
	$(CXX) $(CXXFLAGS) $(COBJS) -o main

FEM.o: FEM.cu FEM.h
	$(CXX) $(CXXFLAGS) FEM.cu -c

ShapeFunctions.o: ShapeFunctions.cu ShapeFunctions.h
	$(CXX) $(CXXFLAGS) ShapeFunctions.cu -c

Quadtree.o: Quadtree.cu Quadtree.h
	$(CXX) $(CXXFLAGS) Quadtree.cu -c

run:
	./main

clean:
	rm -rf *.o
	rm main
