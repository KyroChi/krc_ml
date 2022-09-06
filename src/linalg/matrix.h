#ifndef MATRIX_H
#define MATRIX_H

typedef enum {
	NONE,
	MATRIX_BAD_ACCESS,
	MATRIX_SINGULAR,
} mat_err;

// mat_err MATRIX_ERROR = NONE;

typedef struct {
/** 
 * Store data in rows than columns
 */
	unsigned int n_rows;
	unsigned int n_cols;
	double* data;
} Matrix2D;

unsigned int matrix_is_square(Matrix2D*);
double kd (unsigned int, unsigned int);
Matrix2D* zeros (unsigned int, unsigned int);
Matrix2D* ones (unsigned int, unsigned int);
Matrix2D* empty_matrix_2D (unsigned int r_dim, unsigned int c_dim);
void free_matrix_2D (Matrix2D* A);
int check_index_validity (Matrix2D* A, unsigned int i, unsigned int j, char*);
double index_matrix_2D (Matrix2D* A, unsigned int i, unsigned int j);
void print_matrix (Matrix2D* A);
void print_matrixf (Matrix2D* A, unsigned int);
void print_array (double* dptr, unsigned int length);
void set_matrix_2D (Matrix2D* A, unsigned int i, unsigned int j,
		    double d);
void get_col (Matrix2D* A, unsigned int c, double* col);
void get_row (Matrix2D* A, unsigned int c, double* row);
Matrix2D* matmul (Matrix2D* A, Matrix2D* B);

Matrix2D* matrix_map (Matrix2D*, double (*map_fun)(double));
void _matrix_map_subroutine (Matrix2D*, Matrix2D*, double (*map_fun)(double));
double identity (double);
Matrix2D* matrix_copy (Matrix2D*);
Matrix2D* matrix_map_param (Matrix2D*,
			    double (*map_fun)(double, double*),
			    double*);
Matrix2D* zip_matrix_map (Matrix2D*, Matrix2D*, double (*map_fun)(double, double));
double add (double, double);
double sub (double, double);
double mul (double, double);
double mul_param (double, double*);
Matrix2D* matadd (Matrix2D*, Matrix2D*);
Matrix2D* matsub (Matrix2D*, Matrix2D*);
Matrix2D* had_prod (Matrix2D*, Matrix2D*);
Matrix2D* scalar_mult (Matrix2D*, double);

Matrix2D* lu_decomp (Matrix2D*);
void lu_decomp_inplace (Matrix2D*);

Matrix2D* lu_backsub (Matrix2D*, Matrix2D*);
Matrix2D* eye (unsigned int);

Matrix2D* matinv (Matrix2D* );
Matrix2D* transpose (Matrix2D*);
void transpose_inplace (Matrix2D*);

Matrix2D* slice (Matrix2D*,
		 unsigned int, unsigned int,
		 unsigned int, unsigned int);

void dimension_mismatch (Matrix2D*, Matrix2D*, char*);
Matrix2D* concatenate (Matrix2D*, Matrix2D*, unsigned int);

double matrix_cumsum (Matrix2D*);

#endif
