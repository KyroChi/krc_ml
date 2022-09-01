#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "matrix.h"

/* #define PARALLEL */

int
main (int argc, char **argv)
{
	Matrix2D* mat = empty_matrix_2D(2, 2);
	double farr[4] = {1,2,3,4};
	mat->data = farr;
	/* double ind = index_matrix_2D(mat, 1, 1); */
	/* print_matrix(mat); */
	/* set_matrix_2D(mat, 1, 1, 6); */
	/* print_matrix(mat); */
	/* double rc[2]; */
	/* get_row(mat, 1, rc); */
	/* print_array(rc, 2); */
	/* get_col(mat, 0, rc); */
	/* print_array(rc, 2); */
	/* printf("---------\n"); */
	/* print_matrix(matmul(mat, mat)); */

	/* print_matrix(mat); */
	/* print_matrix(matadd(mat, mat)); */
	/* print_matrix(matsub(mat, mat)); */
	/* print_matrix(had_prod(mat, mat)); */

	/* printf("---------\n"); */
	/* set_matrix_2D(mat, 1, 1, 4); */
	/* print_matrix(mat); */
	/* printf("\n"); */
	/* lu_decomp_inplace(mat); */
	/* print_matrix(mat); */

	Matrix2D* I2 = eye(2);

	Matrix2D* LU = lu_decomp(mat);
	Matrix2D* sol = lu_backsub(LU, I2);

	print_matrix(sol);

	Matrix2D* inv = matinv(mat);
       
	print_matrix(inv);

	printf("\n");

	mat = empty_matrix_2D(3, 3);
	double farr2[9] = {1,2,2,3,1,-1,1,1,-2};
	mat->data = farr2;

	print_matrix(mat);
	printf("\n");

	free(inv);
	inv = matinv(mat);
	print_matrix(inv);
	
}

double
kd (unsigned int row, unsigned int col)
{
	if (row == col) {
		return 1.0;
	} else {
		return 0.0;
	}
}

Matrix2D*
empty_matrix_2D (unsigned int n_rows, unsigned int n_cols)
{

	Matrix2D* mat_ptr;
	mat_ptr = malloc(sizeof(*mat_ptr));
	
	mat_ptr->n_rows = n_rows;
	mat_ptr->n_cols = n_cols;
	mat_ptr->data = malloc(sizeof(double) * n_rows * n_cols);
	return mat_ptr;
}

int
check_index_validity (Matrix2D* A, unsigned int i,
		      unsigned int j)
{
	if ((i > A->n_rows - 1) || (j > A->n_cols - 1)) {
		printf("Attempt to index into a matrix failed.\n"
		       "A is dimension (%d, %d). Attempted index "
		       "(%d, %d).\n\n", A->n_rows, A->n_cols, i, j);
		MATRIX_ERROR = MATRIX_BAD_ACCESS;
		return 0;
	} else {
		return 1;
	}
}

double
index_matrix_2D (Matrix2D* A, unsigned int i,
		 unsigned int j)
/**
 * Zero indexes into the matrix 
 *
 * Returns -1 if the indexing went poorly.
 */
{
	if (check_index_validity(A, i, j)) {
		return *(A->data + i * A->n_rows + j);
	} else {
		// TODO: Handle this error better.
		return -1;
	}
}

void
set_matrix_2D (Matrix2D* A, unsigned int i, unsigned int j, double d)
{
	if (check_index_validity(A, i, j)) {
		*(A->data + i * A->n_rows + j) = d;
	} 
}

void
print_matrix (Matrix2D* A)
{
	double num = 0;
	printf("[");
	for (int ii = 0; ii < A->n_rows; ii += 1) {
		if (ii == 0) {
			printf("[ ");
		} else {
			printf(" [ ");
		}
		
		for (int jj = 0; jj < A->n_cols; jj += 1) {
			num = index_matrix_2D(A, ii, jj);
			printf("%.2f \t",  num);
		}
		
		if (ii < A->n_rows - 1) {
			printf(" ],\n");
		}
	}
	printf(" ]]\n");
	return;
}

void
print_array (double* dptr, unsigned int length)
{
	for (int ii = 0; ii < length; ii += 1) {
		printf("%.2f ", *(dptr + ii));
	}
	
	printf("\n");
	return;
}

void
get_row (Matrix2D* A, unsigned int r, double* row)
/**
 * Pass a pre-allocated double* to take the values. This is so we
 * don't have to repeatedly malloc and free during matrix
 * multiplications.
 *
 * Thus... we assume that row is sufficiently long already!
 */
{
	for (int ii = 0; ii < A->n_cols; ii += 1) {
		*(row + ii) = *(A->data + r * A->n_rows + ii);
	}
}

void
get_col (Matrix2D* A, unsigned int c, double* col)
{
	for (int ii = 0; ii < A->n_rows; ii += 1) {
		*(col + ii) = *(A->data + ii * A->n_rows + c);
	}
}

double
dotprod (double* a, double* b, unsigned int length)
{
	double sum = 0;
	
	for (int ii = 0; ii < length; ii += 1) {
		sum += ( *(a + ii) * *(b + ii) );
	}
	
	return sum;
}

Matrix2D*
matrix_map (Matrix2D* A, double (*map_fun)(double))
/**
 * Map a function onto a matrix and return a new matrix.
 */
{
	Matrix2D* res = empty_matrix_2D(A->n_rows, A->n_cols);
	double a;

	for (int ii = 0; ii < A->n_rows; ii += 1) {
		for (int jj = 0; jj < A->n_cols; jj += 1) {
			a = index_matrix_2D(A, ii, jj);
			set_matrix_2D(res, ii, jj, map_fun(a));
		}
	}

	return res;
}

double identity (double a) { return a; }

Matrix2D*
matrix_copy (Matrix2D* A)
{
	return matrix_map(A, &identity);
}

Matrix2D*
matrix_map_param (Matrix2D* A,
		  double (*map_fun)(double, double),
		  double param)
{
	Matrix2D* res = empty_matrix_2D(A->n_rows, A->n_cols);
	double a;

	for (int ii = 0; ii < A->n_rows; ii += 1) {
		for (int jj = 0; jj < A->n_cols; jj += 1) {
			a = index_matrix_2D(A, ii, jj);
			set_matrix_2D(res, ii, jj, map_fun(a, param));
		}
	}

	return res;
}

Matrix2D*
scalar_mult (Matrix2D* A, double a)
{
	return matrix_map_param(A, &mul, a);
}


Matrix2D*
zip_matrix_map (Matrix2D* A,
		Matrix2D* B,
		double (*map_fun)(double, double))
/**
 * Map a function taking two doubles on the zipped matrices (A, B).
 * I.e. computes AB[i][j] = fun(A[i][j], B[i][j]).
 */
{
	// Verify matching dimensions
	if ((A->n_rows != B->n_rows) || (A->n_cols != B->n_cols)) {
		printf("Addition: Incompatible dimensions (%d, %d)"
		       " and (%d, %d).\n",
		       A->n_rows, A->n_cols, B->n_rows, B->n_cols);
		return NULL;
	}

	Matrix2D* res = empty_matrix_2D(A->n_rows, A->n_cols);
	double a;
	double b;

	for (int ii = 0; ii < A->n_rows; ii += 1) {
		for (int jj = 0; jj < A->n_cols; jj += 1) {
			a = index_matrix_2D(A, ii, jj);
			b = index_matrix_2D(B, ii, jj);
			set_matrix_2D(res, ii, jj, map_fun(a, b));
		}
	}

	return res;
}

double add (double a, double b) { return a + b; }
double sub (double a, double b) { return a - b; }
double mul (double a, double b) { return a * b; }

Matrix2D*
matadd (Matrix2D* A, Matrix2D* B)
{
	return zip_matrix_map(A, B, &add);
}

Matrix2D*
matsub (Matrix2D* A, Matrix2D* B)
{
	return zip_matrix_map(A, B, &sub);
}

Matrix2D*
had_prod (Matrix2D* A, Matrix2D* B)
{
	return zip_matrix_map(A, B, &mul);
}

Matrix2D*
matmul (Matrix2D* A, Matrix2D* B)
/*
 * Compute the matrix product of the 2D matrices A and B.
 */
{
	if (A->n_cols != B->n_rows) {
		printf("Incompatible dimensions (%d, %d) and (%d, %d).\n",
		       A->n_rows, A->n_cols, B->n_rows, B->n_cols);
		return NULL;
	}

	Matrix2D* AB = empty_matrix_2D(A->n_rows, B->n_cols);

	double row[A->n_cols];
	double col[B->n_rows];
	double dot = 0;

	pthread_t threads[B->n_cols];

	for (int ii = 0; ii < A->n_rows; ii += 1) {
		for (int jj = 0; jj < B->n_cols; jj += 1) {
			/* This loop can be run in parallel */
#ifndef PARALLEL
			get_row(A, ii, row);
			get_col(B, jj, col);
			dot = dotprod(row, col, A->n_cols);
			set_matrix_2D(AB, ii, jj, dot);
#endif
#ifdef PARALLEL
			get_row(A, ii, row);
			get_col(B, jj, col);
			/* pthread_create(&threads[jj], NULL, */
			/* 	       dotprod,  */
#endif
		}
	}
	
	return AB;
}

double
determinant (Matrix2D* A)
/** 
 * Compute the determinant of the matrix A
 */
{
	// Check if square
	return 0;
}

unsigned int
matrix_is_square (Matrix2D* A)
{
	if (A->n_rows == A->n_cols) {
		return 1;
	} else {
		return 0;
	}
}

Matrix2D*
lu_decomp (Matrix2D* A)
/**
 * Compute the LU decomposition of the matrix A using Crout's
 * algorithm.
 *
 * We take L = [a_{ij}], a_{ij} = 0 if i > j and a_{ii} = 1.
 * U = [b_{ij}], b_{ij} = 0 if i < j. We return the matrix
 * M = L + U - I, i.e. M = m_{ij} where m_{ij} = a_{ij} if i < j and
 * m_{ij} = b_{ij} if i >= j.
 *
 *
 * See Press, Teukolsku, Betterling, Flannery Chapter 2.3.
 */
{
	if (!matrix_is_square(A)) {
		// TODO: error messagee
		return NULL;
	}

	Matrix2D* Acpy = matrix_copy(A);
	lu_decomp_inplace(Acpy);
	return Acpy;
}

void
lu_decomp_inplace (Matrix2D* A)
/**
 * Preforms Crout's algorithm in-place on a matrix A.
 */
{
	double a, alphaij, alphaik, betaij, betakj, sum;
	for (int jj = 0; jj < A->n_rows; jj += 1) {
		for (int ii = 0; ii <= jj; ii +=1) {
			a = index_matrix_2D(A, ii, jj);

			sum = 0;
			for (int kk = 0; kk < ii; kk += 1) {
				alphaik = index_matrix_2D(A, ii, kk);
				betakj = index_matrix_2D(A, kk, jj);
				sum += alphaik * betakj;
			}
			
			betaij = a - sum;

			set_matrix_2D(A, ii, jj, betaij);
		}

		for (int ii = jj + 1; ii < A->n_rows; ii +=1) {
			a = index_matrix_2D(A, ii, jj);

			sum = 0;
			for (int kk = 0; kk < jj; kk += 1) {
				if (ii == kk) {
					alphaik = 1;
				} else {
					alphaik = index_matrix_2D(A,
								  ii,
								  kk);
				}
				betakj = index_matrix_2D(A, kk, jj);

				sum += alphaik * betakj;
			}
			
			betaij = index_matrix_2D(A, jj, jj);
			alphaij = 1/betaij * (a - sum);

			set_matrix_2D(A, ii, jj, alphaij);
		}
	}

	return;
}

Matrix2D*
eye (unsigned int size) {
	Matrix2D* eye = empty_matrix_2D(size, size);

	for (int ii = 0; ii < size; ii += 1) {
		for (int jj = 0; jj < size; jj += 1) {
			if (ii == jj) {
				set_matrix_2D(eye, ii, jj, 1);
			} else {
				set_matrix_2D(eye, ii, jj, 0);
			}
		}
	}

	return eye;
}

Matrix2D*
lu_backsub (Matrix2D* LU, Matrix2D* B)
/**
 * Input LU is expected to be the output from the LU decomp routine.
 *
 * Solves equations of the form, LU X = B. X dimensions is same as B 
 * dimensions.
 *
 * Soves for each columns, so consider (LU)x = b for column vectors 
 * x, b. We solve Ly = b for y. Then Ux = y for x. These linear 
 * equations are each easy to see.
 *
 * See Press, Teukolsku, Betterling, Flannery Chapter 2.3. 
 */
{
	// TODO: Check that the dimensions line up

	Matrix2D* X = empty_matrix_2D(B->n_rows, B->n_cols);
	Matrix2D* y = empty_matrix_2D(B->n_rows, 1);

	double yj, yk, xj, xk, bj, lujj, lujk, sum;

	// Solve each column of B as a linear eq. (LU)x = b
	for (int ii = 0; ii < B->n_cols; ii += 1) {
		for (int jj = 0; jj < B->n_rows; jj += 1) {
			sum = 0;
			for (int kk = 0; kk < jj; kk += 1) {
				lujk = index_matrix_2D(LU, jj, kk);
				yk = index_matrix_2D(y, kk, 0);
				sum += lujk * yk;
			}

			bj = index_matrix_2D(B, jj, ii);
			
			yj = (bj - sum);

			set_matrix_2D(y, jj, 0, yj);
		}

		for (int jj = B->n_rows - 1; jj >= 0; jj -= 1) {
			sum = 0;
			for (int kk = jj; kk < B->n_rows; kk += 1) {
				lujk = index_matrix_2D(LU, jj, kk);
				xk = index_matrix_2D(X, kk, ii);
				sum += lujk * xk;
			}

			lujj = index_matrix_2D(LU, jj, jj);
			
			yj = index_matrix_2D(y, jj, 0);
			xj = 1 / lujj * (yj - sum);

			set_matrix_2D(X, jj, ii, xj);
		}
	}

	free(y);
	return X;
}

Matrix2D*
matinv (Matrix2D* A)
/**
 * Compute the inverse of the square matrix A
 */
{
	// TODO: Check if square
	
	if (A->n_rows == 2) {
		// Faster than running through the algorithm.
		Matrix2D* res = empty_matrix_2D(2, 2);
		double a, b, c, d;
		
		a = index_matrix_2D(A, 0, 0);
		b = index_matrix_2D(A, 0, 1);
		c = index_matrix_2D(A, 1, 0);
		d = index_matrix_2D(A, 1, 1);
		
		set_matrix_2D(res, 0, 0, d);
		set_matrix_2D(res, 0, 1, -b);
		set_matrix_2D(res, 1, 0, -c);
		set_matrix_2D(res, 1, 1, a);
		
		Matrix2D* tmp = scalar_mult(res, 1 / ( a*d - b*c ));
		free(res);
		return tmp;
	}

	// Matrix is not 2 x 2
	Matrix2D* LU = lu_decomp(A);
	Matrix2D* I = eye(A->n_rows);

	Matrix2D* Ainv = lu_backsub(LU, I);

	free(LU);
	free(I);
	
	return Ainv;
}
