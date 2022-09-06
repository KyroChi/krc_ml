#include <Python.h>
#include "matrix.h"

static PyObject* linalg_empty_matrix_2D (PyObject* self, PyObject* args);

static PyMethodDef LinalgMethods[] = {
	{"empty_matrix", linalg_empty_matrix_2D, METH_VARARGS,
	 "Initialize an empty matrix."},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef linalgmodule = {
	PyModuleDef_HEAD_INIT,
	"linalg",
	NULL,
	-1,
	LinalgMethods
};

PyMODINIT_FUNC
PyInit_linalg(void)
{
	return PyModule_Create(&linalgmodule);
}

static PyObject*
linalg_empty_matrix_2D (PyObject* self, PyObject* args)
{
	unsigned int dim[2];
	
	if (!PyArg_ParseTuple(args, "ii", &dim)) {
		return NULL;
	}

	MAT2D* A = empty_matrix_2D(*dim, *(dim + 1));
	print_matrix(A);
	free(A);

	Py_INCREF(Py_None);
	return Py_None;
}
