import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.perlin3d import generate_perlin_noise_3d
import mcubes
import pyvista as pv
import time

np.random.seed(0)
print("Generando ruido Perlin 3D...")
time_start = time.time()
noise = generate_perlin_noise_3d(
    (32, 256, 256), (1, 4, 4), tileable=(True, False, False)
)
time_end = time.time()
print(f"Ruido Perlin 3D generado en {time_end - time_start:.2f} segundos")

fig = plt.figure()
images = [
    [plt.imshow(
        layer, cmap='gray', interpolation='lanczos', animated=True
    )]
    for layer in noise
]
animation_3d = animation.ArtistAnimation(fig, images, interval=50, blit=True)
plt.show()

# 3. Aplicar Marching Cubes sobre el volumen
isovalue = 0.0  # puedes experimentar con este valor
vertices, triangles = mcubes.marching_cubes(noise, isovalue)

# 4. Exportar como archivo OBJ
mcubes.export_obj(vertices, triangles, "perlin_surface.obj")
print("Superficie exportada como 'perlin_surface.obj'")

# 5. Crear malla con PyVista
mesh = pv.PolyData(vertices, np.hstack([
    np.full((triangles.shape[0], 1), 3),  # número de puntos por cara (siempre 3)
    triangles
]).astype(np.int64))  # PyVista requiere enteros de 64 bits

# 6. Visualizar
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.add_axes()
plotter.show_bounds(grid='back', location='outer', all_edges=True)
plotter.show()