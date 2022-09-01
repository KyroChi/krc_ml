typedef enum {
	NONE,
	MATRIX_BAD_ACCESS,
	MATRIX_SINGULAR,
} mat_err;

mat_err MATRIX_ERROR = NONE;

typedef struct {
/** 
 * Store data in rows than columns
 */
	unsigned int n_rows;
	unsigned int n_cols;
	double* data;
} Matrix2D;

typedef struct {
	unsigned int n_rows;
	unsigned int n_cols;
	double* data;
} LowerDiagMatrix2D;

typedef struct {
	unsigned int n_rows;
	unsigned int n_cols;
	double* data;
} UpperDiagMatrix2D;

unsigned int matrix_is_square(Matrix2D*);
double kd (unsigned int, unsigned int);
Matrix2D* empty_matrix_2D (unsigned int r_dim, unsigned int c_dim);
int check_index_validity (Matrix2D* A, unsigned int i, unsigned int j);
double index_matrix_2D (Matrix2D* A, unsigned int i, unsigned int j);
void print_matrix (Matrix2D* A);
void print_array (double* dptr, unsigned int length);
void set_matrix_2D (Matrix2D* A, unsigned int i, unsigned int j,
		    double d);
void get_col (Matrix2D* A, unsigned int c, double* col);
void get_row (Matrix2D* A, unsigned int c, double* row);
Matrix2D* matmul (Matrix2D* A, Matrix2D* B);

Matrix2D* matrix_map (Matrix2D*, double (*map_fun)(double));
double identity (double);
Matrix2D* matrix_copy (Matrix2D*);
Matrix2D* matrix_map_param (Matrix2D*, double (*map_fun)(double, double), double);
Matrix2D* zip_matrix_map (Matrix2D*, Matrix2D*, double (*map_fun)(double, double));
double add (double, double);
double sub (double, double);
double mul (double, double);
Matrix2D* matadd (Matrix2D*, Matrix2D*);
Matrix2D* matsub (Matrix2D*, Matrix2D*);
Matrix2D* had_prod (Matrix2D*, Matrix2D*);
Matrix2D* scalar_mult (Matrix2D*, double);

Matrix2D* lu_decomp (Matrix2D*);
void lu_decomp_inplace (Matrix2D*);

Matrix2D* lu_backsub (Matrix2D*, Matrix2D*);
Matrix2D* eye (unsigned int);

Matrix2D* matinv (Matrix2D* );
