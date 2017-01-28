FLAGS=-std=c++11 -Ofast
LIBS=-I/usr/local/opt/openblas/include/ -lblas

single_machine:
	g++ $(FLAGS) src/single_machine_nn.cpp $(LIBS) -o single_machine

single_machine_run:
	rm -f single_machine
	make single_machine
	./single_machine
