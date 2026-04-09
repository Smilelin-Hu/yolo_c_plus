#include "TrtInfer.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace yolo {
namespace {

void printUsage() {
    std::cout << "Usage: inspect_engine --engine model.engine" << std::endl;
}

bool hasDynamicDims(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

std::string bytesDescription(const TensorInfo& tensor) {
    if (hasDynamicDims(tensor.dims)) {
        return "dynamic";
    }
    const size_t bytes = static_cast<size_t>(volume(tensor.dims)) * elementSize(tensor.dtype);
    std::ostringstream oss;
    oss << bytes;
    return oss.str();
}

std::string dynamicDimsDescription(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    bool first = true;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] >= 0) {
            continue;
        }
        if (!first) {
            oss << ',';
        }
        oss << i;
        first = false;
    }
    if (first) {
        return "static";
    }
    return std::string("dynamic_indices=") + oss.str();
}

}  // namespace
}  // namespace yolo

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            yolo::printUsage();
            return 1;
        }

        std::string engine_path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for " + name);
                }
                return argv[++i];
            };

            if (arg == "--engine") {
                engine_path = next(arg);
            } else if (arg == "--help") {
                yolo::printUsage();
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        if (engine_path.empty()) {
            throw std::runtime_error("--engine is required");
        }

        yolo::TrtEngine engine(engine_path);
        for (const auto& tensor : engine.tensors()) {
            std::cout << (tensor.is_input ? "[input] " : "[output] ") << tensor.name
                      << " is_input=" << (tensor.is_input ? "true" : "false")
                      << " shape=" << yolo::dimsToString(tensor.dims)
                      << " dtype=" << yolo::dataTypeToString(tensor.dtype)
                      << " bytes=" << yolo::bytesDescription(tensor)
                      << " dims=" << yolo::dynamicDimsDescription(tensor.dims) << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "inspect_engine failed: " << e.what() << std::endl;
        return 1;
    }
}
