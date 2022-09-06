#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "initializers.h"
#include "../krc_ml_lib.h"
#include "matrix.h"

/* double */
/* normal_dist (double sigma2, double mu) */
/* /\* Implemented using Box-Muller transformation. Apparently  */
/*    this is not the best algorithm. *\/ */
/* { */
/* 	UNUSED(x); */
/* 	x = rand() */
/* 	return 1 / sqrt( 2 * M_PI * sigma2) * */
/* 		exp( - (x - mu)**2 / (2 * sigma2) ); */
/* } */

/* double */
/* normal_dist_map_wrapper (double x, double *sm) */
/* { */
/* 	UNUSED(x); */

/* 	double sigma2 = sm[0]; */
/* 	double mu     = sm[1]; */

/* 	return normal_dist(sigma2, mu); */
/* } */

/* void */
/* GAUSSIAN_initializer (Matrix2D* mat) */
/* { */
/* 	double params[2] = {1, 0}; */
/* 	matrix_map_param(mat, &normal_dist_map_wrapper, &params); */
/* 	return; */
			 
/* } */

double
UNIFORM (double a) { UNUSED(a); return (double) rand() / RAND_MAX; }

void
UNIFORM_initializer (Matrix2D* mat)
{
	_matrix_map_subroutine(mat, mat, &UNIFORM);
	return;
}

/* void */
/* GLOROT_initializer (Matrix2D* mat) */
/* { */
/* 	double sigma2 = 2 / ( mat->n_rows + mat->n_cols ); */
/* 	double params[2] = {sigma */
/* } */

/* void */
/* HE_initializer (Matrix2D* mat); */
