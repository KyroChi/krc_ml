#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "matrix.h"
#include "../krc_ml_lib.h"

/* #define PARALLEL */

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

void
free_matrix_2D (Matrix2D* A)
{
	free(A->data);
	free(A);
	return;
}

double zero (double a) { UNUSED(a); return 0.0; }
double one (double a) { UNUSED(a); return 1.0; }

Matrix2D*
zeros (unsigned int n_rows, unsigned int n_cols)
{
	Matrix2D* z = empty_matrix_2D(n_rows, n_cols);
	_matrix_map_subroutine(z, z, &zero);
	return z;
}

Matrix2D*
ones (unsigned int n_rows, unsigned int n_cols)
{
	Matrix2D* z = empty_matrix_2D(n_rows, n_cols);
	_matrix_map_subroutine(z, z, &one);
	return z;
}

int
check_index_validity (Matrix2D* A, unsigned int i,
		      unsigned int j, char* caller)
{
	if ((i > A->n_rows - 1) || (j > A->n_cols - 1)) {
		printf("%s: Attempt to index into a matrix failed.\n"
		       "A is dimension (%d, %d). Attempted index "
		       "(%d, %d).\n\n",
		       caller, A->n_rows, A->n_cols, i, j);
		// MATRIX_ERROR = MATRIX_BAD_ACCESS;
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
	if (check_index_validity(A, i, j, "index_matrix_2D")) {
		return *(A->data + i * A->n_rows + j);
	} else {
		// TODO: Handle this error better.
		return -1;
	}
}

void
set_matrix_2D (Matrix2D* A, unsigned int i, unsigned int j, double d)
{
	if (check_index_validity(A, i, j, "set_matrix_2D")) {
		*(A->data + i * A->n_rows + j) = d;
	} 
}

void
print_matrixf (Matrix2D* A, unsigned int precision)
{
	double num = 0;
	printf("[");
	unsigned int ii, jj;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		if (ii == 0) {
			printf("[ ");
		} else {
			printf(" [ ");
		}
		
		for (jj = 0; jj < A->n_cols - 1; jj += 1) {
			num = index_matrix_2D(A, ii, jj);
			printf("%.*f \t",  precision, num);
		}

		num = index_matrix_2D(A, ii, jj);
		printf("%.*f",  precision, num);
		
		if (ii < A->n_rows - 1) {
			printf(" ],\n");
		}
	}
	printf(" ]]\n");
	return;
}

void
print_matrix (Matrix2D* A)
{
	print_matrixf(A, 2);
}

void
print_array (double* dptr, unsigned int length)
{
	unsigned int ii;
	for (ii = 0; ii < length; ii += 1) {
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
	unsigned int ii;
	for (ii = 0; ii < A->n_cols; ii += 1) {
		*(row + ii) = *(A->data + r * A->n_rows + ii);
	}
}

void
get_col (Matrix2D* A, unsigned int c, double* col)
{
	unsigned int ii;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		*(col + ii) = *(A->data + ii * A->n_rows + c);
	}
}

double
dotprod (double* a, double* b, unsigned int length)
{
	double sum = 0;

	unsigned int ii;
	for (ii = 0; ii < length; ii += 1) {
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
	
	_matrix_map_subroutine(A, res, map_fun);

	return res;
}

void
_matrix_map_subroutine (Matrix2D* in, Matrix2D* out,
			double (*map_fun)(double))
/**
 * Set in and out to the same matrix for inplace.
 */
{
	// TODO: Make this function safe.
	double a;
	unsigned int ii, jj;
	for (ii = 0; ii < in->n_rows; ii += 1) {
		for (jj = 0; jj < in->n_cols; jj += 1) {
			a = index_matrix_2D(in, ii, jj);
			set_matrix_2D(out, ii, jj, map_fun(a));
		}
	}
	
}

double identity (double a) { return a; }

Matrix2D*
matrix_copy (Matrix2D* A)
{
	return matrix_map(A, &identity);
}

Matrix2D*
matrix_map_param (Matrix2D* A,
		  double (*map_fun)(double, double*),
		  double *params)
{
	Matrix2D* res = empty_matrix_2D(A->n_rows, A->n_cols);
	double a;

	unsigned int ii, jj;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		for (jj = 0; jj < A->n_cols; jj += 1) {
			a = index_matrix_2D(A, ii, jj);
			set_matrix_2D(res, ii, jj, map_fun(a, params));
		}
	}

	return res;
}

Matrix2D*
scalar_mult (Matrix2D* A, double a)
{
	return matrix_map_param(A, &mul_param, &a);
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

	unsigned int ii, jj;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		for (jj = 0; jj < A->n_cols; jj += 1) {
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
double mul_param (double a, double *b) { return a * *b; }

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

	unsigned int ii, jj;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		for (jj = 0; jj < B->n_cols; jj += 1) {
			get_row(A, ii, row);
			get_col(B, jj, col);
			dot = dotprod(row, col, A->n_cols);
			set_matrix_2D(AB, ii, jj, dot);
		}
	}
	
	return AB;
}

/* double */
/* determinant (Matrix2D* A) */
/* /\**  */
/*  * Compute the determinant of the matrix A */
/*  *\/ */
/* { */
/* 	// Check if square */
/* 	return 0; */
/* } */

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
	
	unsigned int ii, jj, kk;
	for (jj = 0; jj < A->n_rows; jj += 1) {
		for (ii = 0; ii <= jj; ii +=1) {
			a = index_matrix_2D(A, ii, jj);

			sum = 0;
			for (kk = 0; kk < ii; kk += 1) {
				alphaik = index_matrix_2D(A, ii, kk);
				betakj = index_matrix_2D(A, kk, jj);
				sum += alphaik * betakj;
			}
			
			betaij = a - sum;

			set_matrix_2D(A, ii, jj, betaij);
		}

		for (ii = jj + 1; ii < A->n_rows; ii +=1) {
			a = index_matrix_2D(A, ii, jj);

			sum = 0;
			for (kk = 0; kk < jj; kk += 1) {
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

	unsigned int ii, jj;
	for (ii = 0; ii < size; ii += 1) {
		for (jj = 0; jj < size; jj += 1) {
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
	unsigned int ii, jj, kk, mm;
	for (ii = 0; ii < B->n_cols; ii += 1) {
		for (jj = 0; jj < B->n_rows; jj += 1) {
			sum = 0;
			for (kk = 0; kk < jj; kk += 1) {
				lujk = index_matrix_2D(LU, jj, kk);
				yk = index_matrix_2D(y, kk, 0);
				sum += lujk * yk;
			}

			bj = index_matrix_2D(B, jj, ii);
			
			yj = (bj - sum);

			set_matrix_2D(y, jj, 0, yj);
		}

		for (mm = 0; mm < B->n_rows; mm += 1) {
			jj = B->n_rows - 1 - mm;
			sum = 0;

			for (kk = jj; kk < B->n_rows; kk += 1) {
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

	if (A->n_rows != A->n_cols) {
		printf("Cannot invert non-square matrix.\n");
		return NULL;
	}

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
		free_matrix_2D(res);
		return tmp;
	}

	// Matrix is not 2 x 2
	Matrix2D* LU = lu_decomp(A);
	Matrix2D* I = eye(A->n_rows);

	Matrix2D* Ainv = lu_backsub(LU, I);

	free_matrix_2D(LU);
	free_matrix_2D(I);
	
	return Ainv;
}

Matrix2D*
transpose (Matrix2D* A)
// TODO: Write an in-place version of this function, I think it should
// be faster...
{
	Matrix2D* AT = empty_matrix_2D(A->n_cols, A->n_rows);

	unsigned int ii, jj;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		for (jj = 0; jj < A->n_cols; jj += 1) {
			set_matrix_2D(AT, jj, ii,
				      index_matrix_2D(A, ii, jj));
		}
	}

	return AT;
}

void
transpose_inplace (Matrix2D* A)
{
	if (A->n_rows != A->n_cols) {
		dimension_mismatch(A, A, "tranpose in-place");
		return;
	}

	double tmp;

	unsigned int ii, jj;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		for (jj = 0; jj < A->n_cols; jj += 1) {
			tmp = index_matrix_2D(A, ii, jj);
			set_matrix_2D(A, ii, jj,
				      index_matrix_2D(A, jj, ii));
			set_matrix_2D(A, jj, ii, tmp);
		}
	}

	return;
}

void
dimension_mismatch (Matrix2D* A, Matrix2D* B, char* operation)
{
	printf("Incompatible dimensions (%d, %d) and (%d, %d)"
	       " for operation %s.\n",
	       A->n_rows, A->n_cols, B->n_rows, B->n_cols,
	       operation);
	return;
}

Matrix2D*
concatenate (Matrix2D* A, Matrix2D* B, unsigned int axis)
// Since only rank 2 tensors only two axis choices
{
	Matrix2D* concat;
	double entry;
	
	if (axis == 0) {
		if (A->n_rows != B->n_rows) {
			dimension_mismatch(A, B,
					   "concatenate on axis 0");
			return NULL;
		}

		concat = empty_matrix_2D(A->n_rows,
					 A->n_cols + B->n_cols);

		unsigned int ii, jj;
		for (ii = 0; ii < A->n_rows; ii += 1) {
			for (jj = 0; jj < A->n_cols; jj += 1) {
				entry = index_matrix_2D(A, ii, jj);
				set_matrix_2D(concat, ii, jj, entry);
			}

			for (jj = 0; jj < B->n_cols; jj += 1) {
				entry = index_matrix_2D(B, ii, jj);
				set_matrix_2D(concat, ii,
					      A->n_cols + jj, entry);
			}
		}

		return concat;
	} else if (axis == 1) {
		if (A->n_cols != B->n_cols) {
			dimension_mismatch(A, B,
					   "concatenate on axis 1");
			return NULL;
		}

		concat = empty_matrix_2D(A->n_rows + B->n_rows,
					 A->n_cols);

		unsigned int ii, jj;
		for (jj = 0; jj < A->n_cols; jj += 1) {
			for (ii = 0; ii < A->n_rows; ii += 1) {
				entry = index_matrix_2D(A, ii, jj);
				set_matrix_2D(concat, ii, jj, entry);
			}

			for (ii = 0; ii < B->n_rows; ii += 1) {
				entry = index_matrix_2D(B, ii, jj);
				set_matrix_2D(concat, A->n_rows + ii,
					      jj, entry);
			}
		}

		return concat;
	}

	printf("Attempt to concatenate along unknown axis "
	       "%d.", axis);
	return NULL;
}

Matrix2D*
slice (Matrix2D* A,
       unsigned int row_start, unsigned int row_stop,
       unsigned int col_start, unsigned int col_stop)
{
	if ( (row_start > row_stop) || (col_start > col_stop) ) {
		printf("Attempted slice with bad indicies. "
		       "Row and column indicies must be ordered.\n");
	}

	if ( (row_stop >= A->n_rows) || (col_stop >= A->n_cols) ) {
		printf("Attempted slice with bad indicies. "
		       "Row and column indicies must be less than"
		       " dimensions of matrix.\n");
	}
	
	unsigned int n_rows = row_stop - row_start + 1;
	unsigned int n_cols = col_stop - col_start + 1;

	double entry;
	Matrix2D* S = empty_matrix_2D(n_rows, n_cols);

	unsigned int ii, jj;
	for (ii = 0; ii < n_rows; ii += 1) {
		for (jj = 0; jj < n_cols; jj += 1) {
			entry = index_matrix_2D(A,
						row_start + ii,
						col_start + jj);
			set_matrix_2D(S, ii, jj, entry);
		}
	}

	return S;
}

double
matrix_cumsum (Matrix2D* A)
/* Sum all the elements of a matrix. */
{
	double sum = 0;
	
	unsigned int ii, jj;
	for (ii = 0; ii < A->n_rows; ii += 1) {
		for (jj = 0; jj < A->n_cols; jj += 1) {
			sum += index_matrix_2D(A, ii, jj);
		}
	}

	return sum;
}
