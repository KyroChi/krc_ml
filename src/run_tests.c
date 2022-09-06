#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "linalg/matrix.h"
#include "ml/least_squares.h"
#include "run_tests.h"

int
main (/*int argc, char **argv*/)
{
	/* Matrix2D* mat = empty_matrix_2D(2, 2); */
	/* double farr[4] = {1,2,3,4}; */
	/* mat->data = farr; */
	/* /\* double ind = index_matrix_2D(mat, 1, 1); *\/ */
	/* /\* print_matrix(mat); *\/ */
	/* /\* set_matrix_2D(mat, 1, 1, 6); *\/ */
	/* /\* print_matrix(mat); *\/ */
	/* /\* double rc[2]; *\/ */
	/* /\* get_row(mat, 1, rc); *\/ */
	/* /\* print_array(rc, 2); *\/ */
	/* /\* get_col(mat, 0, rc); *\/ */
	/* /\* print_array(rc, 2); *\/ */
	/* /\* printf("---------\n"); *\/ */
	/* /\* print_matrix(matmul(mat, mat)); *\/ */

	/* /\* print_matrix(mat); *\/ */
	/* /\* print_matrix(matadd(mat, mat)); *\/ */
	/* /\* print_matrix(matsub(mat, mat)); *\/ */
	/* /\* print_matrix(had_prod(mat, mat)); *\/ */

	/* /\* printf("---------\n"); *\/ */
	/* /\* set_matrix_2D(mat, 1, 1, 4); *\/ */
	/* /\* print_matrix(mat); *\/ */
	/* /\* printf("\n"); *\/ */
	/* /\* lu_decomp_inplace(mat); *\/ */
	/* /\* print_matrix(mat); *\/ */

	/* Matrix2D* Z = zeros(4, 4); */
	/* print_matrix(Z); */
	/* printf("\n"); */

	/* Matrix2D* I2 = eye(2); */

	/* Matrix2D* LU = lu_decomp(mat); */
	/* Matrix2D* sol = lu_backsub(LU, I2); */

	/* print_matrix(sol); */

	/* Matrix2D* inv = matinv(mat); */
       
	/* print_matrix(inv); */

	/* Matrix2D* invT = transpose(inv); */
	/* print_matrix(invT); */

	/* Matrix2D* C = concatenate(inv, invT, 0); */
	/* print_matrix(C); */
	/* printf("\n"); */
	/* print_matrix(concatenate(inv, invT, 1)); */
	/* printf("slice %d, %d\n", C->n_rows, C->n_cols); */
	/* print_matrix(slice(C, 0, 0, 0, 3)); */

	/* printf("\n"); */

	/* mat = empty_matrix_2D(3, 3); */
	/* double farr2[9] = {1,2,2,3,1,-1,1,1,-2}; */
	/* mat->data = farr2; */

	/* print_matrix(mat); */
	/* printf("\n"); */

	/* free(inv); */
	/* inv = matinv(mat); */
	/* print_matrix(inv); */
	unsigned int passed = 0;
	unsigned int failed = 0;

	run_matrix_tests(&passed, &failed);
	run_least_squares_tests(&passed, &failed);

	printf("Passed: %d\nFailed: %d\nTotal:  %d\n------------\n",
	       passed, failed, passed + failed);
}


void run_matrix_tests (unsigned int* passed, unsigned int* failed)
{
	Matrix2D* mat = NULL;
	Matrix2D* inv = NULL;
	
	/* Test 2x2 inverse */
	mat = empty_matrix_2D(2, 2);
	mat->data[0] = 1;
	mat->data[1] = 2;
	mat->data[2] = 3;
	mat->data[3] = 4;

	inv = matinv(mat);

	mat->data[0] = -2;
	mat->data[1] = 1;
	mat->data[2] = 1.5;
	mat->data[3] = -0.5;

	double any = matrix_cumsum(matsub(mat, inv));

	if (any == 0) {
		*passed += 1;
	} else {
		printf("Failed 2x2 matrix inverse.\n");
		*failed += 1;
	}

	free_matrix_2D(mat);
	free_matrix_2D(inv);

	/* Test 3x3 inverse */
	mat = empty_matrix_2D(3, 3);
	mat->data[0] = 1;
	mat->data[1] = 0;
	mat->data[2] = 0;
	mat->data[3] = 0;
	mat->data[4] = 1;
	mat->data[5] = 0;
	mat->data[6] = 0;
	mat->data[7] = 0;
	mat->data[8] = 1;

	inv = lu_decomp(mat);
	any = matrix_cumsum(matsub(mat, inv));

	if (any == 0) {
		*passed += 1;
	} else {
		printf("Failed 3x3 matrix inverse test 1.\n");
		*failed += 1;
	}

	free_matrix_2D(mat);
	free_matrix_2D(inv);

	/* Test 3x3 inverse */
	mat = empty_matrix_2D(3, 3);
	mat->data[0] = 1.0;
	mat->data[1] = 2.0;
	mat->data[2] = -1.0;
	mat->data[3] = 2.0;
	mat->data[4] = 1.0;
	mat->data[5] = 2.0;
	mat->data[6] = -1.0;
	mat->data[7] = 2.0;
	mat->data[8] = 1.0;
	
	inv = matinv(mat);

	mat->data[0] = 0.1875;
	mat->data[1] = 0.25;
	mat->data[2] = -0.3125;
	mat->data[3] = 0.25;
	mat->data[4] = 0;
	mat->data[5] = 0.25;
	mat->data[6] = -0.3125;
	mat->data[7] = 0.25;
	mat->data[8] = 0.1875;

	// TODO: The nested matsub leaks memory.
	any = matrix_cumsum(matsub(mat, inv));

	if (any == 0) {
		*passed += 1;
	} else {
		printf("Failed 3x3 matrix inverse test 2.\n");
		*failed += 1;
	}

	// TODO: These frees cause a seg fault.
	/* free_matrix_2D(mat); */
	/* free_matrix_2D(inv); */
}

void run_least_squares_tests (unsigned int* passed,
			      unsigned int* failed)
{
	Matrix2D* beta = NULL;
	Matrix2D* bias = NULL;
	Matrix2D* X = empty_matrix_2D(3, 2);
	Matrix2D* y = empty_matrix_2D(3, 1);

	X->data[0] = 0;
	X->data[1] = -1/3;
	X->data[2] = 1/2;
	X->data[3] = 1/2 + 1/3;
	X->data[4] = 1;
	X->data[5] = 2/3;

	y->data[0] = 0;
	y->data[1] = 1;
	y->data[0] = 0;

	least_squares_fit(X, y, beta, bias);
	/* print_matrix(beta); */
	/* print_matrix(bias); */

	*passed += 0;
	*failed += 0;
}
