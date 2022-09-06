#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "nn.h"
#include "linalg/initializers.h"

double TANH (double x);
double TANH (double x)
{
	return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

int
main ()
{
	SequentialNet *net = NULL;
	unsigned int n_hidden = 4;
	unsigned int widths[4] = {4, 2, 3, 4};

	net = new_dense_nn(3, 2, n_hidden, widths);

	Matrix2D *X = empty_matrix_2D(3, 1);
	X->data[0] = 1;
	X->data[1] = 1;
	X->data[2] = 1;

	unsigned int ii;
	for (ii = 0; ii < net->n_layers + 1; ii += 1) {
		printf("Weight: %d, (%d, %d)\n",
		       ii,
		       net->weights[ii]->n_rows,
		       net->weights[ii]->n_cols);
		print_matrixf(net->weights[ii], 5);
	}
	printf("\n");

	Matrix2D *yhat = feed_forward(net, X);
	print_matrixf(yhat, 5);
	
	return 0;
}

SequentialNet *
new_dense_nn (unsigned int input_dimension,
	      unsigned int output_dimension,
	      unsigned int n_layers,          // Number of hidden layers
	      unsigned int *layer_widths)     // Must be n_layers long
{
	SequentialNet *net = malloc(sizeof(SequentialNet));
	
	net->n_layers         = n_layers;
	net->input_dimension  = input_dimension;
	net->output_dimension = output_dimension;

	Matrix2D **weights =
		malloc( sizeof(Matrix2D*) * (n_layers + 1) );

	weights[0] =
		empty_matrix_2D(layer_widths[0], input_dimension);
	UNIFORM_initializer(weights[0]);

	unsigned int ii;
	for (ii = 0; ii < n_layers - 1; ii += 1) {
		weights[ii + 1] =
			empty_matrix_2D(layer_widths[ii + 1],
					layer_widths[ii]);
		UNIFORM_initializer(weights[ii + 1]);
	}
	
	weights[ii + 1] = empty_matrix_2D(output_dimension,
					  layer_widths[ii]);
	UNIFORM_initializer(weights[ii + 1]);

	net->weights = weights;

	return net;
}

Matrix2D *
feed_forward (SequentialNet *net,
	      Matrix2D *X)
{
	// TODO: Do dimension checks!
	Matrix2D *tmp1 = matrix_copy(X);
	Matrix2D *tmp2 = NULL;
	
	unsigned int ii;
	for (ii = 0; ii < net->n_layers + 1; ii += 1) {
		/* TODO: This is horribly leaky. */
		tmp2 = matmul(net->weights[ii], tmp1);
		// free_matrix_2D(tmp1);
		tmp1 = matrix_map(tmp2, &TANH);
		// free_matrix_2D(tmp2);
	}

	return tmp1;
}

FeedForwardOutput *
new_ffout (SequentialNet *net, unsigned int training)
{
	FeedForwardOutput *out = malloc(sizeof(FeedForwardOutput));

	out->training = training;
	out->net = net;

	if (training) {
		out->activations = malloc(sizeof(Matrix2D*) *
					  (net->n_layers + 1));
		out->local_derivatives = malloc(sizeof(Matrix2D*) *
						(net->n_layers + 1));
	} else {
		out->activations = NULL;
		out->local_derivatives = NULL;
	}

	return out;
}

void
free_ffout (FeedForwardOutput *ffout)
/**
 * Frees memory for yhat, activations, and local derivatives, but NOT 
 * net.
 */
{
	if (training) {
		// Free the activations and local_derivatives
	} // else: the activations and local_derivatives are NULL.

	free_matrix_2D(yhat);
	free(ffout);
	
	return;
}

FeedForwardOutput *
_feed_forward (SequentialNet *net,
	       Matrix2D *X,
	       unsigned int training)
/**
 * Return a list of Matrix2D's:
 *    0:
 *        The output from the feed-forward.
 *    1 - n_layers+1:
 *        The activations at the (i - 1)th layer.
 *    n_layers+2 - 2*(n_layers+1):
 *        The local derivatives at the (i - n_layers)th layer.
 */
{
	unsigned int ii;
	unsigned int output_length = 2*(net->n_layers + 1) + 1;
	FeedForwardOutput *out = new_ffout(net);
	
	if (training) {
	        outputs = malloc( sizeof(Matrix2D*) * output_length );
		
		outputs[0] =
			empty_matrix_2D(net->output_dimension, 1);
		
		for (ii = 1; ii < net->n_layers; ii += 1) {
			outputs[ii];
			outputs[ii + n_layers];
		}
	}
	// We can allocate all of the sizes beforeehand
	
	Matrix2D *tmp1 = matrix_copy(X);
	Matrix2D *tmp2 = NULL;
	
	for (ii = 0; ii < net->n_layers + 1; ii += 1) {
		/* TODO: This is horribly leaky. */
		tmp2 = matmul(net->weights[ii], tmp1);
		// free_matrix_2D(tmp1);
		tmp1 = matrix_map(tmp2, &TANH);
		// free_matrix_2D(tmp2);

		if (training) {
			// We can set to tmp1 and avoid a leak since
			// we assign a new pointer to tmp1 in the next
			// loop.
			out->activations[ii] = tmp1;
		}
	}

	out->yhat = tmp1;

	return out;
}

/* void */
/* train_nn (SequentialNet* net, */
/* 	  Matrix2D* X, */
/* 	  Matrix2D* y, */
/* 	  unsigned int epochs) */
/* { */
/* 	return; */
/* } */

/* void */
/* initalize_weights (SequentialNet* net) */
/* { */
/* } */


/* GO EVEN SIMPLER: Feed forward densely connected network, hardcoded */
