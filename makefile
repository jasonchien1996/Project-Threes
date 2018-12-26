all:
	g++ -std=c++11 -O3 -fopenmp -g -Wall -fmessage-length=0 -o main main.cpp
clean:
	rm main
