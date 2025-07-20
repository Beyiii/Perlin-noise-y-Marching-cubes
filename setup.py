from skbuild import setup

setup(
    name="mc_cuda",
    version="0.1",
    description="Marching Cubes using CUDA",
    cmake_install_dir="mc_cuda",  # Esto debe coincidir con el DESTINATION
    packages=["mc_cuda"],
    python_requires=">=3.7",
)