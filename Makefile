FLAGS=

single_machine:
	g++ $(FLAGS) src/single_machine_nn.cpp -o single_machine

single_machine_run:
	rm -f single_machine
	make single_machine
	./single_machine
