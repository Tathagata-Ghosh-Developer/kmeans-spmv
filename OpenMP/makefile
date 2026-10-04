CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native
LDFLAGS := -fopenmp

all: kmeans_sequential kmeans_parallel

kmeans_sequential: kmeans_sequential.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

kmeans_parallel: kmeans_parallel.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

clean:
	rm -f kmeans_sequential kmeans_parallel *.o
