#ifndef PYWRAPPER_CUDA_H
#define PYWRAPPER_CUDA_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#endif

PyObject* marching_cubes_cuda_wrapper(PyArrayObject* arr, double isolevel);

#ifdef __cplusplus
}
#endif

#endif // PYWRAPPER_CUDA_H
