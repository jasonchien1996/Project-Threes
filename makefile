all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes 2048.cpp
clean:
	rm threes
