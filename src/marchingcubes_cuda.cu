#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "marchingcubes_cuda.h"
#include "tables.h" 

#define MAX_TRIANGLES_PER_VOXEL 5

__device__ float3 interpolate(float3 p1, float3 p2, float valp1, float valp2, float isolevel) {
    if (fabs(isolevel - valp1) < 0.00001)
        return p1;
    if (fabs(isolevel - valp2) < 0.00001)
        return p2;
    if (fabs(valp1 - valp2) < 0.00001)
        return p1;
    float mu = (isolevel - valp1) / (valp2 - valp1);
    return make_float3(
        p1.x + mu * (p2.x - p1.x),
        p1.y + mu * (p2.y - p1.y),
        p1.z + mu * (p2.z - p1.z)
    );
}

__global__ void marching_kernel(
    const float* data, int xdim, int ydim, int zdim,
    float isolevel,
    float3* vertices, int3* triangles,
    int* vertex_count, int* triangle_count
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= xdim - 1 || y >= ydim - 1 || z >= zdim - 1)
        return;

    int idx = x + y * xdim + z * xdim * ydim;

    float cube[8];
    float3 p[8];

    int dx = 1;
    int dy = xdim;
    int dz = xdim * ydim;

    cube[0] = data[idx];
    cube[1] = data[idx + dx];
    cube[2] = data[idx + dx + dy];
    cube[3] = data[idx + dy];
    cube[4] = data[idx + dz];
    cube[5] = data[idx + dx + dz];
    cube[6] = data[idx + dx + dy + dz];
    cube[7] = data[idx + dy + dz];

    p[0] = make_float3(x,     y,     z);
    p[1] = make_float3(x + 1, y,     z);
    p[2] = make_float3(x + 1, y + 1, z);
    p[3] = make_float3(x,     y + 1, z);
    p[4] = make_float3(x,     y,     z + 1);
    p[5] = make_float3(x + 1, y,     z + 1);
    p[6] = make_float3(x + 1, y + 1, z + 1);
    p[7] = make_float3(x,     y + 1, z + 1);

    int cubeindex = 0;
    if (cube[0] < isolevel) cubeindex |= 1;
    if (cube[1] < isolevel) cubeindex |= 2;
    if (cube[2] < isolevel) cubeindex |= 4;
    if (cube[3] < isolevel) cubeindex |= 8;
    if (cube[4] < isolevel) cubeindex |= 16;
    if (cube[5] < isolevel) cubeindex |= 32;
    if (cube[6] < isolevel) cubeindex |= 64;
    if (cube[7] < isolevel) cubeindex |= 128;

    int edges = edgeTable[cubeindex];
    if (edges == 0) return;

    float3 vertlist[12];

    if (edges & 1)
        vertlist[0] = interpolate(p[0], p[1], cube[0], cube[1], isolevel);
    if (edges & 2)
        vertlist[1] = interpolate(p[1], p[2], cube[1], cube[2], isolevel);
    if (edges & 4)
        vertlist[2] = interpolate(p[2], p[3], cube[2], cube[3], isolevel);
    if (edges & 8)
        vertlist[3] = interpolate(p[3], p[0], cube[3], cube[0], isolevel);
    if (edges & 16)
        vertlist[4] = interpolate(p[4], p[5], cube[4], cube[5], isolevel);
    if (edges & 32)
        vertlist[5] = interpolate(p[5], p[6], cube[5], cube[6], isolevel);
    if (edges & 64)
        vertlist[6] = interpolate(p[6], p[7], cube[6], cube[7], isolevel);
    if (edges & 128)
        vertlist[7] = interpolate(p[7], p[4], cube[7], cube[4], isolevel);
    if (edges & 256)
        vertlist[8] = interpolate(p[0], p[4], cube[0], cube[4], isolevel);
    if (edges & 512)
        vertlist[9] = interpolate(p[1], p[5], cube[1], cube[5], isolevel);
    if (edges & 1024)
        vertlist[10] = interpolate(p[2], p[6], cube[2], cube[6], isolevel);
    if (edges & 2048)
        vertlist[11] = interpolate(p[3], p[7], cube[3], cube[7], isolevel);

    for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
        int v_idx[3];
        for (int j = 0; j < 3; ++j) {
            int edge = triTable[cubeindex][i + j];
            int idx = atomicAdd(vertex_count, 1);
            vertices[idx] = vertlist[edge];
            v_idx[j] = idx;
        }

        int t_idx = atomicAdd(triangle_count, 1);
        triangles[t_idx] = make_int3(v_idx[0], v_idx[1], v_idx[2]);
    }
}

void launch_marching_cubes(
    const float* d_data, int xdim, int ydim, int zdim,
    float isolevel,
    float3* d_vertices, int3* d_triangles,
    int* d_vertex_count, int* d_triangle_count
) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(
        (xdim + threadsPerBlock.x - 2) / (threadsPerBlock.x),
        (ydim + threadsPerBlock.y - 2) / (threadsPerBlock.y),
        (zdim + threadsPerBlock.z - 2) / (threadsPerBlock.z)
    );

    marching_kernel<<<numBlocks, threadsPerBlock>>>(
        d_data, xdim, ydim, zdim,
        isolevel,
        d_vertices, d_triangles,
        d_vertex_count, d_triangle_count
    );
    cudaDeviceSynchronize();
}

Mesh marching_cubes_cuda(float* h_data, int xdim, int ydim, int zdim, float isolevel) {
    int volume_size = xdim * ydim * zdim;
    int max_voxels = (xdim - 1) * (ydim - 1) * (zdim - 1);
    int max_triangles = max_voxels * MAX_TRIANGLES_PER_VOXEL;
    int max_vertices = max_triangles * 3;

    // Device pointers
    float* d_data;
    float3* d_vertices;
    int3* d_triangles;
    int* d_vertex_count;
    int* d_triangle_count;

    cudaMalloc(&d_data, sizeof(float) * volume_size);
    cudaMalloc(&d_vertices, sizeof(float3) * max_vertices);
    cudaMalloc(&d_triangles, sizeof(int3) * max_triangles);
    cudaMalloc(&d_vertex_count, sizeof(int));
    cudaMalloc(&d_triangle_count, sizeof(int));

    cudaMemcpy(d_data, h_data, sizeof(float) * volume_size, cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpy(d_vertex_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangle_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(8, 8, 8);
    dim3 blocks((xdim + threads.x - 2)/ (threads.x),
                (ydim + threads.y - 2)/ (threads.y),
                (zdim + threads.z - 2)/ (threads.z));

    marching_kernel<<<blocks, threads>>>(
        d_data, xdim, ydim, zdim, isolevel,
        d_vertices, d_triangles, d_vertex_count, d_triangle_count
    );

    cudaDeviceSynchronize();

    int h_vertex_count = 0, h_triangle_count = 0;
    cudaMemcpy(&h_vertex_count, d_vertex_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_triangle_count, d_triangle_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Copiar resultados a host
    float3* h_vertices3 = new float3[h_vertex_count];
    int3* h_faces3 = new int3[h_triangle_count];
    cudaMemcpy(h_vertices3, d_vertices, sizeof(float3) * h_vertex_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_faces3, d_triangles, sizeof(int3) * h_triangle_count, cudaMemcpyDeviceToHost);

    // Empaquetar resultado en Mesh
    Mesh mesh;
    mesh.num_vertices = h_vertex_count * 3;
    mesh.num_faces = h_triangle_count * 3;
    mesh.vertices = new float[mesh.num_vertices];
    mesh.faces = new int[mesh.num_faces];

    for (int i = 0; i < h_vertex_count; ++i) {
        mesh.vertices[3 * i + 0] = h_vertices3[i].x;
        mesh.vertices[3 * i + 1] = h_vertices3[i].y;
        mesh.vertices[3 * i + 2] = h_vertices3[i].z;
    }

    for (int i = 0; i < h_triangle_count; ++i) {
        mesh.faces[3 * i + 0] = h_faces3[i].x;
        mesh.faces[3 * i + 1] = h_faces3[i].y;
        mesh.faces[3 * i + 2] = h_faces3[i].z;
    }

    // Liberar memoria temporal
    delete[] h_vertices3;
    delete[] h_faces3;

    cudaFree(d_data);
    cudaFree(d_vertices);
    cudaFree(d_triangles);
    cudaFree(d_vertex_count);
    cudaFree(d_triangle_count);

    return mesh;
}

// También definimos esto para facilitar la liberación desde Python
void free_mesh(Mesh mesh) {
    delete[] mesh.vertices;
    delete[] mesh.faces;
}