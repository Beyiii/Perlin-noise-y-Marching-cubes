#ifndef MARCHINGCUBES_CUDA_H
#define MARCHINGCUBES_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

struct Mesh {
float* vertices; // (x, y, z)
int num_vertices;
int* faces; // (a, b, c)
int num_faces;
};

Mesh marching_cubes_cuda(float* data, int xdim, int ydim, int zdim, float isolevel);

// Función para liberar la memoria de un Mesh devuelto
void free_mesh(Mesh mesh);

#ifdef __cplusplus
}
#endif

#endif 