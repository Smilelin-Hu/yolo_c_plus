#pragma once

#include "Types.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace yolo {

class TrtLogger final : public nvinfer1::ILogger {
public:
    explicit TrtLogger(Severity severity = Severity::kINFO) : severity_(severity) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > severity_) {
            return;
        }
        std::cerr << "[TensorRT] " << msg << std::endl;
    }

private:
    Severity severity_;
};

template <typename T>
struct TrtDestroy {
    void operator()(T* ptr) const {
        if (ptr != nullptr) {
            delete ptr;
        }
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroy<T>>;

inline void throwIfCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ':' << line << " for " << expr << ": "
            << cudaGetErrorString(code);
        throw std::runtime_error(oss.str());
    }
}

#define YOLO_CUDA_CHECK(expr) ::yolo::throwIfCudaError((expr), #expr, __FILE__, __LINE__)

inline int64_t volume(const nvinfer1::Dims& dims) {
    int64_t value = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        value *= std::max<int64_t>(dims.d[i], 1);
    }
    return value;
}

inline size_t elementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
        default:
            throw std::runtime_error("Unsupported TensorRT data type");
    }
}

inline std::string dataTypeToString(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT8:
            return "int8";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kBOOL:
            return "bool";
        case nvinfer1::DataType::kUINT8:
            return "uint8";
        default:
            return "unknown";
    }
}

inline nvinfer1::Dims toDims(const std::vector<int>& values) {
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(values.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = values[static_cast<size_t>(i)];
    }
    return dims;
}

inline std::vector<TensorInfo> collectTensorInfo(nvinfer1::ICudaEngine& engine) {
    std::vector<TensorInfo> tensors;
    const int num_io = engine.getNbIOTensors();
    tensors.reserve(static_cast<size_t>(num_io));
    for (int i = 0; i < num_io; ++i) {
        const char* name = engine.getIOTensorName(i);
        TensorInfo info;
        info.name = name != nullptr ? name : "";
        info.dims = engine.getTensorShape(info.name.c_str());
        info.dtype = engine.getTensorDataType(info.name.c_str());
        info.is_input = engine.getTensorIOMode(info.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        tensors.push_back(info);
    }
    return tensors;
}

inline std::string dimsToString(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << '[';
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i != 0) {
            oss << 'x';
        }
        oss << dims.d[i];
    }
    oss << ']';
    return oss.str();
}

}  // namespace yolo
