# Makefile for MPI SpMV
CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -Wall

TARGET = spmv_mpi
SOURCES = main_mpi.cpp sparse_matrix.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

# Helper to run a quick test on 8 cores (assuming running on a node directly or via salloc)
test_local: $(TARGET)
	mpiexec -np 8 ./$(TARGET)