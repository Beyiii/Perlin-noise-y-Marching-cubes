#include "pywrapper_cuda.h"
#include "marchingcubes_cuda.h"  // tu header CUDA con struct Mesh y función marching_cubes_cuda
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

PyObject* marching_cubes_cuda_wrapper(PyArrayObject* arr, double isolevel)
{
    try {
        if (!arr) {
            PyErr_SetString(PyExc_ValueError, "Input array is NULL");
            return NULL;
        }

        if (PyArray_NDIM(arr) != 3) {
            PyErr_SetString(PyExc_ValueError, "Input array must be 3D");
            return NULL;
        }

        if (PyArray_TYPE(arr) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_TypeError, "Input array must be of type float32");
            return NULL;
        }

        int xdim = (int)PyArray_DIM(arr, 0);
        int ydim = (int)PyArray_DIM(arr, 1);
        int zdim = (int)PyArray_DIM(arr, 2);

        float* h_data = (float*)PyArray_DATA(arr);
        size_t vol_size = (size_t)xdim * ydim * zdim;

        // Reserva memoria device para el volumen
        float* d_data = nullptr;
        cudaMalloc((void**)&d_data, vol_size * sizeof(float));
        cudaMemcpy(d_data, h_data, vol_size * sizeof(float), cudaMemcpyHostToDevice);

        // Llama a la función CUDA (marching cubes)
        Mesh mesh = marching_cubes_cuda(d_data, xdim, ydim, zdim, (float)isolevel);

        // Reserva buffers host para resultados
        size_t num_verts = mesh.num_vertices;
        size_t num_faces = mesh.num_faces;

        float* h_vertices = new float[num_verts * 3];
        int* h_faces = new int[num_faces * 3];

        // Copia resultados de device a host
        cudaMemcpy(h_vertices, mesh.vertices, num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_faces, mesh.faces, num_faces * 3 * sizeof(int), cudaMemcpyDeviceToHost);

        // Crea arrays numpy para vertices y faces
        npy_intp dims_vertices[2] = {(npy_intp)num_verts, 3};
        PyObject* vertices_arr = PyArray_SimpleNew(2, dims_vertices, NPY_FLOAT);

        npy_intp dims_faces[2] = {(npy_intp)num_faces, 3};
        PyObject* faces_arr = PyArray_SimpleNew(2, dims_faces, NPY_INT);

        memcpy(PyArray_DATA((PyArrayObject*)vertices_arr), h_vertices, num_verts * 3 * sizeof(float));
        memcpy(PyArray_DATA((PyArrayObject*)faces_arr), h_faces, num_faces * 3 * sizeof(int));

        // Limpieza
        delete[] h_vertices;
        delete[] h_faces;
        cudaFree(d_data);
        cudaFree(mesh.vertices);
        cudaFree(mesh.faces);

        return Py_BuildValue("(OO)", vertices_arr, faces_arr);
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
