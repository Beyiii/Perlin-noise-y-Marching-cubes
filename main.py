import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.perlin3d import generate_perlin_noise_3d
import mcubes
import pyvista as pv
import time
import mc_cuda

# 1. Creación de un volumen de ruido Perlin 3D
np.random.seed(0)
print("Generando ruido Perlin 3D...")
time_start = time.time()
noise = generate_perlin_noise_3d(
    (32, 256, 256), (1, 4, 4), tileable=(True, False, False)
)
time_end = time.time()
print(f"Ruido Perlin 3D generado en {time_end - time_start:.2f} segundos")

# 2. Visualización del ruido Perlin 3D
fig = plt.figure()
images = [
    [plt.imshow(
        layer, cmap='gray', interpolation='lanczos', animated=True
    )]
    for layer in noise
]
animation_3d = animation.ArtistAnimation(fig, images, interval=50, blit=True)
plt.show()

isovalue = 0.0  # puedes experimentar con este valor
vertices, triangles = mc_cuda.marching_cubes_cuda(noise, isovalue)
print(f"Vertices shape: {vertices.shape}")
print(f"Faces shape: {triangles.shape}")

"""
# 3. Aplicar Marching Cubes sobre el volumen
print("Aplicando Marching Cubes...")
time_start = time.time()
isovalue = 0.0  # puedes experimentar con este valor
vertices, triangles = mcubes.marching_cubes(noise, isovalue)
time_end = time.time()
print(f"Marching Cubes aplicado en {time_end - time_start:.2f} segundos")

# 4. Crear malla con PyVista
mesh = pv.PolyData(vertices, np.hstack([
    np.full((triangles.shape[0], 1), 3),  # número de puntos por cara (siempre 3)
    triangles
]).astype(np.int64))  # PyVista requiere enteros de 64 bits

# 5. Visualizar
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.add_axes()
plotter.show_bounds(grid='back', location='outer', all_edges=True)
plotter.show()

"""