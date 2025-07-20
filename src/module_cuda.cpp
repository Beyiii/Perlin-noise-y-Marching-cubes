#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "pywrapper_cuda.h"

static PyObject* py_marching_cubes_cuda(PyObject* self, PyObject* args)
{
    PyObject* input_obj = nullptr;
    double isovalue;

    if (!PyArg_ParseTuple(args, "Od", &input_obj, &isovalue)) {
        return NULL;
    }

    PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!arr) return NULL;

    PyObject* result = marching_cubes_cuda_wrapper(arr, isovalue);
    Py_DECREF(arr);

    return result;
}

static PyMethodDef methods[] = {
    {"marching_cubes_cuda", py_marching_cubes_cuda, METH_VARARGS, "Run Marching Cubes on GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mc_cuda",  // módulo python que importarás
    "Marching Cubes CUDA Module",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_mc_cuda(void)
{
    import_array(); // Inicializa numpy C API
    return PyModule_Create(&moduledef);
}
