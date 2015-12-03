a.out: main.o nn.o
	g++ -o a.out main.o nn.o

main.o: 
	g++ -c main.cpp

nn.o: nn.h
	g++ -c nn.cpp

cn:
	rm *.o *.out *~ *.train* *.result*
