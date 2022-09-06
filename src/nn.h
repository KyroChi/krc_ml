/* Move to other files soon */

#ifndef NN_H
#define NN_H

#include "linalg/matrix.h"

#define UNUSED(x) do { (void)(x); } while (0)

typedef struct {
	unsigned int n_layers;
	unsigned int input_dimension;
	unsigned int output_dimension;
	Matrix2D** weights;
} SequentialNet;

typedef struct {
	unsigned int training;
	SequentialNet *net;
	Matrix2D *yhat;
	Matrix2D **activations;
	Matrix2D **local_derivatives;
} FeedForwardOutput;

FeedForwardOutput *new_ffout(SequentialNet *, unsigned int);

SequentialNet *new_dense_nn (unsigned int,
			     unsigned int,
			     unsigned int,
			     unsigned int*);
Matrix2D *feed_forward (SequentialNet *net, Matrix2D *X);

#endif
