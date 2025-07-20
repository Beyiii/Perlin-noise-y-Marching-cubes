cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0146 NEW)
project(mc_cuda LANGUAGES CXX CUDA)

# Detectar Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Encontrar CUDA
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
find_package(CUDAToolkit REQUIRED)

# Obtener ruta a los includes de numpy desde Python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Incluir directorios para numpy y código fuente
include_directories(${NUMPY_INCLUDE_DIR})
include_directories(src)

# Archivos fuente
set(SOURCES
    src/module_cuda.cpp
    src/pywrapper_cuda.cpp
    src/marchingcubes_cuda.cu
)

# Crear el módulo
add_library(mc_cuda SHARED ${SOURCES})

set_target_properties(mc_cuda PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    CUDA_SEPARABLE_COMPILATION ON
)

# Flags para CUDA (opcionales)
target_compile_options(mc_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

# Vincular con cudart
target_link_libraries(mc_cuda PRIVATE CUDA::cudart Python3::Python)

# Para que se construya en mc_cuda/ como un paquete
install(TARGETS mc_cuda DESTINATION mc_cuda)  # ✅ Instala dentro del proyecto

