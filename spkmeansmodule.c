#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"
#include "kmeans.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>


/*
 * Helper Methods
 */
PyObject* double_matrix_to_pylist(double** mat, int rows, int cols) {
    int i, j;
    PyObject* pylist = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(row, j, Py_BuildValue("d", mat[i][j]));
        }
        PyList_SetItem(pylist, i, row);
    }
    return pylist;
}

static PyObject* fit_helper(int k, int max_iter, double eps, char* tmp_combined_inputs, char* tmp_initial_centroids) {
    double** centroids;
    PyObject* py_centroids;
    centroids = K_means(k, max_iter, eps, tmp_combined_inputs, tmp_initial_centroids);
    py_centroids = double_matrix_to_pylist(centroids, k, count_cols(tmp_initial_centroids));
    return py_centroids;
}

static PyObject* create_wam_helper(int n, int d, char* input_file) {
    double** W;
    PyObject* py_W;
    W = create_wam_api(n, d, input_file); /* Call function from spkmeans.c*/
    py_W = double_matrix_to_pylist(W, n, n);
    return py_W;
}

static PyObject* create_ddg_helper(int n, int d, char* input_file) {
    double** D;
    PyObject* py_D;
    D = create_ddg_api(n, d, input_file); /* Call function from spkmeans.c*/
    py_D = double_matrix_to_pylist(D, n, n);
    return py_D;
}

static PyObject* create_lnorm_helper(int n, int d, char* input_file) {
    double** lnorm;
    PyObject* py_lnorm;
    lnorm = create_lnorm_api(n, d, input_file); /* Call function from spkmeans.c*/
    py_lnorm = double_matrix_to_pylist(lnorm, n, n);
    return py_lnorm;
}

static PyObject* create_jacobi_helper(int n, int d, char* input_file) {
    double** jacobi;
    PyObject* py_jacobi;
    jacobi = create_jacobi_api(n, d, input_file); /* Call function from spkmeans.c*/
    py_jacobi = double_matrix_to_pylist(jacobi, n+1, n);
    return py_jacobi;
}

static void spk_capi_helper(int k, char* input_file) {
    /* calls spk_api from spkmeans.c, which given a file path
     * calculates T and writes it to a temporary file
     * */
    spk_api(k, input_file);
}


/*
 * API Methods to be called from python
 */

static PyObject* fit_capi(PyObject *self, PyObject *args){
    int K;
    int max_iter;
    double epsilon;
    char* tmp_combined_inputs;
    char* tmp_initial_centroids;
    /* Parse Python Object into C variabels */
    if (!PyArg_ParseTuple(args,"iidss",&K, &max_iter, &epsilon, &tmp_combined_inputs,&tmp_initial_centroids)){
        return NULL;
    }
    /* call fit_helper helper function, build python object from the return value */
    return Py_BuildValue("O", fit_helper(K, max_iter, epsilon, tmp_combined_inputs, tmp_initial_centroids));
}

static PyObject* wam_capi(PyObject *self, PyObject *args){
    int n;
    int d;
    char* input_file;
    /* Parse Python Object into C variabels */
    if (!PyArg_ParseTuple(args,"iis",&n, &d, &input_file)){
        return NULL;
    }
    /* call fit_helper helper function, build python object from the return value */
    return Py_BuildValue("O", create_wam_helper(n, d, input_file));
}

static PyObject* ddg_capi(PyObject *self, PyObject *args){
    int n;
    int d;
    char* input_file;
    /* Parse Python Object into C variabels */
    if (!PyArg_ParseTuple(args,"iis",&n, &d, &input_file)){
        return NULL;
    }
    /* call fit_helper helper function, build python object from the return value */
    return Py_BuildValue("O", create_ddg_helper(n, d, input_file));
}

static PyObject* lnorm_capi(PyObject *self, PyObject *args){
    int n;
    int d;
    char* input_file;
    /* Parse Python Object into C variabels */
    if (!PyArg_ParseTuple(args,"iis",&n, &d, &input_file)){
        return NULL;
    }
    /* call fit_helper helper function, build python object from the return value */
    return Py_BuildValue("O", create_lnorm_helper(n, d, input_file));
}

static PyObject* jacobi_capi(PyObject *self, PyObject *args){
    int n;
    int d;
    char* input_file;
    /* Parse Python Object into C variabels */
    if (!PyArg_ParseTuple(args,"iis",&n, &d, &input_file)){
        return NULL;
    }
    /* call fit_helper helper function, build python object from the return value */
    return Py_BuildValue("O", create_jacobi_helper(n, d, input_file));
}

static void spk_capi(PyObject *self, PyObject *args){
    int k;
    char* input_file;
    /* Parse Python Object into C variabels */
    if (!PyArg_ParseTuple(args,"is", &k, &input_file)){
        printf("An Error Has Occurred");
    }
    /* call fit_helper helper function, build python object from the return value */
    spk_capi_helper(k, input_file);
}


/*
 * Configurations Methods
 */

static PyMethodDef capiMethods[] = {
/*      PythonName      C-Function Name             args presentation   description     */
        {"fit",         (PyCFunction)fit_capi,      METH_VARARGS,       PyDoc_STR("Runs the kmeans algorithm")},
        {"wam",         (PyCFunction)wam_capi,      METH_VARARGS,       PyDoc_STR("creates wam matrix")},
        {"ddg",         (PyCFunction)ddg_capi,      METH_VARARGS,       PyDoc_STR("creates ddg matrix")},
        {"lnorm",       (PyCFunction)lnorm_capi,    METH_VARARGS,       PyDoc_STR("creates lnorm matrix")},
        {"jacobi",      (PyCFunction)jacobi_capi,   METH_VARARGS,       PyDoc_STR("creates jacobi matrix")},
        {"spk",         (PyCFunction)spk_capi,      METH_VARARGS,       PyDoc_STR("calculates T and writes it to a temp file")},
        {NULL,          NULL,                       0,                  NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "spkmeansmodule",
        NULL,
        -1,
        capiMethods
};

PyMODINIT_FUNC PyInit_spkmeansmodule(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}