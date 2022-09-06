CC=cc

FLAGS = -Wall -Werror -Wextra -pedantic
DEBUG = -g

LINALG = 	src/linalg/matrix.c \
		src/linalg/initializers.c

ML = 		src/ml/least_squares.c

TESTS = 	src/run_tests.c

test: $(OBJECTS)
	$(CC) $(TESTS) $(LINALG) $(ML) -o test.o $(DEBUG)

nn_test: src/nn.c
	cc src/nn.c $(LINALG) -o nn_test.o $(DEBUG) $(FLAGS)
