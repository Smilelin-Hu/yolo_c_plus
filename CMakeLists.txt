cmake_minimum_required(VERSION 3.16)

cmake_policy(SET CMP0074 NEW)

project(yolo_trt_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

set(CUDA_ROOT "/usr/local/cuda-12.8")
set(TensorRT_ROOT "/home/ps/train/HJL/TensorRT-10.15.1.29")

option(YOLO_BUILD_EXAMPLES "Build CLI examples" ON)
option(YOLO_BUILD_TESTS "Build tests" OFF)
option(YOLO_ENABLE_FP16 "Enable FP16 build path" ON)
option(YOLO_ENABLE_INT8 "Enable INT8 build path" ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDART_LIB "${CUDA_ROOT}/lib64/libcudart.so")
set(CUBLAS_LIB "${CUDA_ROOT}/lib64/libcublas.so")
set(TENSORRT_INCLUDE_DIR "${TensorRT_ROOT}/include")
set(TENSORRT_LIB "${TensorRT_ROOT}/lib/libnvinfer.so")
set(TENSORRT_ONNX_LIB "${TensorRT_ROOT}/lib/libnvonnxparser.so")
set(TENSORRT_PLUGIN_LIB "${TensorRT_ROOT}/lib/libnvinfer_plugin.so")
set(TENSORRT_BUILDER_RESOURCE_LIB "${TensorRT_ROOT}/lib/libnvinfer_builder_resource_sm86.so.10.15.1")

set(CMAKE_BUILD_RPATH "${TensorRT_ROOT}/lib;${CUDA_ROOT}/lib64")
set(CMAKE_INSTALL_RPATH "${TensorRT_ROOT}/lib;${CUDA_ROOT}/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN FALSE)

if (NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found under ${TENSORRT_INCLUDE_DIR}")
endif()

if (NOT EXISTS "${TENSORRT_LIB}" OR NOT EXISTS "${TENSORRT_ONNX_LIB}" OR NOT EXISTS "${TENSORRT_PLUGIN_LIB}" OR NOT EXISTS "${TENSORRT_BUILDER_RESOURCE_LIB}")
    message(FATAL_ERROR "TensorRT libraries not found under ${TensorRT_ROOT}/lib")
endif()

add_library(yolo_trt_core
    src/BuildEngine.cpp
    src/TrtInfer.cpp
    src/Preprocess.cpp
    src/PreprocessCuda.cu
    src/PostprocessDetect.cu
    src/YoloMeta.cpp
    src/DetectCli.cpp
    src/ImageIO.cpp
    src/VisualizeDetect.cpp
)

target_include_directories(yolo_trt_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDA_ROOT}/include
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_trt_core
    PUBLIC
        ${CUDART_LIB}
        ${CUBLAS_LIB}
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIB}
        ${TENSORRT_ONNX_LIB}
        ${TENSORRT_PLUGIN_LIB}
        ${TENSORRT_BUILDER_RESOURCE_LIB}
        ${OpenCV_LIBS}
)

target_compile_definitions(yolo_trt_core
    PUBLIC
        $<$<BOOL:${YOLO_ENABLE_FP16}>:YOLO_ENABLE_FP16=1>
        $<$<BOOL:${YOLO_ENABLE_INT8}>:YOLO_ENABLE_INT8=1>
)

set_target_properties(yolo_trt_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_RPATH "${TensorRT_ROOT}/lib;${CUDA_ROOT}/lib64;${TENSORRT_BUILDER_RESOURCE_LIB}"
    INSTALL_RPATH "${TensorRT_ROOT}/lib;${CUDA_ROOT}/lib64;${TENSORRT_BUILDER_RESOURCE_LIB}"
)

if (YOLO_BUILD_EXAMPLES)
    add_executable(build_engine tools/BuildEngineMain.cpp)
    target_link_libraries(build_engine PRIVATE yolo_trt_core)

    add_executable(inspect_engine tools/InspectEngineMain.cpp)
    target_link_libraries(inspect_engine PRIVATE yolo_trt_core)

    add_executable(detect_infer DetectInfer.cpp)
    target_link_libraries(detect_infer PRIVATE yolo_trt_core)
endif()
