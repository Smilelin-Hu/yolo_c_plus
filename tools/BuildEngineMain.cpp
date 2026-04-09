#include "BuildEngine.hpp"
#include "Types.hpp"
#include "YoloMeta.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo {
namespace {

PrecisionMode parsePrecision(const std::string& value) {
    if (value == "fp32") {
        return PrecisionMode::kFP32;
    }
    if (value == "fp16") {
        return PrecisionMode::kFP16;
    }
    if (value == "int8") {
        return PrecisionMode::kINT8;
    }
    throw std::runtime_error("Unsupported precision: " + value);
}

TaskType parseTask(const std::string& value) {
    if (value == "detect") {
        return TaskType::kDetection;
    }
    if (value == "classify") {
        return TaskType::kClassification;
    }
    if (value == "segment") {
        return TaskType::kSegmentation;
    }
    throw std::runtime_error("Unsupported task: " + value);
}

std::vector<int> parseShape(const std::string& text) {
    std::vector<int> dims;
    std::string token;
    for (char ch : text) {
        if (ch == 'x' || ch == 'X') {
            if (!token.empty()) {
                dims.push_back(std::stoi(token));
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty()) {
        dims.push_back(std::stoi(token));
    }
    if (dims.empty()) {
        throw std::runtime_error("Invalid shape: " + text);
    }
    return dims;
}

bool parseOnOff(const std::string& name, const std::string& value) {
    if (value == "on") {
        return true;
    }
    if (value == "off") {
        return false;
    }
    throw std::runtime_error("Unsupported value for " + name + ": " + value + " (expected on|off)");
}

void printUsage() {
    std::cout << "Usage: build_engine --onnx model.onnx --engine model.engine --task detect|classify|segment "
                 "[--preset yolo11m|yolo26m] [--precision fp32|fp16|int8] [--workspace-mb 1024] [--tf32 on|off] "
                 "[--min-shape 1x3x320x320] [--opt-shape 1x3x640x640] [--max-shape 1x3x1280x1280] [--num-classes 80] "
                 "[--dla-core 0] [--no-gpu-fallback] [--verbose] [--calibration-cache cache.bin]"
              << std::endl;
}

std::string precisionToString(PrecisionMode mode) {
    switch (mode) {
        case PrecisionMode::kFP32:
            return "fp32";
        case PrecisionMode::kFP16:
            return "fp16";
        case PrecisionMode::kINT8:
            return "int8";
    }
    return "unknown";
}

}  // namespace
}  // namespace yolo

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            yolo::printUsage();
            return 1;
        }

        std::string onnx_path;
        std::string engine_path;
        std::string task_name = "detect";
        std::string preset;
        std::string precision_name = "fp16";
        int num_classes = 80;
        std::vector<int> min_shape{1, 3, 640, 640};
        std::vector<int> opt_shape{1, 3, 2048, 2048};
        std::vector<int> max_shape{4, 3, 4096, 4096};
        size_t workspace_mb = 1024;
        int dla_core = -1;
        bool allow_gpu_fallback = true;
        bool verbose = false;
        bool tf32 = true;
        std::string calibration_cache_path;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for " + name);
                }
                return argv[++i];
            };

            if (arg == "--onnx") {
                onnx_path = next(arg);
            } else if (arg == "--engine") {
                engine_path = next(arg);
            } else if (arg == "--task") {
                task_name = next(arg);
            } else if (arg == "--preset") {
                preset = next(arg);
            } else if (arg == "--precision") {
                precision_name = next(arg);
            } else if (arg == "--num-classes") {
                num_classes = std::stoi(next(arg));
            } else if (arg == "--min-shape") {
                min_shape = yolo::parseShape(next(arg));
            } else if (arg == "--opt-shape") {
                opt_shape = yolo::parseShape(next(arg));
            } else if (arg == "--max-shape") {
                max_shape = yolo::parseShape(next(arg));
            } else if (arg == "--workspace-mb") {
                workspace_mb = static_cast<size_t>(std::stoull(next(arg)));
            } else if (arg == "--tf32") {
                tf32 = yolo::parseOnOff(arg, next(arg));
            } else if (arg == "--dla-core") {
                dla_core = std::stoi(next(arg));
            } else if (arg == "--no-gpu-fallback") {
                allow_gpu_fallback = false;
            } else if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--calibration-cache") {
                calibration_cache_path = next(arg);
            } else if (arg == "--help") {
                yolo::printUsage();
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        if (onnx_path.empty() || engine_path.empty()) {
            throw std::runtime_error("--onnx and --engine are required");
        }
        if (workspace_mb == 0) {
            throw std::runtime_error("--workspace-mb must be greater than 0");
        }

        yolo::ModelMeta meta;
        if (!preset.empty()) {
            meta = yolo::makeMetaFromPreset(preset, num_classes);
        } else {
            meta.task = yolo::parseTask(task_name);
            meta.num_classes = num_classes;
        }

        yolo::BuildConfig config;
        config.onnx_path = onnx_path;
        config.engine_path = engine_path;
        config.meta = meta;
        config.precision = yolo::parsePrecision(precision_name);
        config.workspace_size = workspace_mb * 1024ULL * 1024ULL;
        config.min_shape = std::move(min_shape);
        config.opt_shape = std::move(opt_shape);
        config.max_shape = std::move(max_shape);
        config.dla_core = dla_core;
        config.verbose = verbose;
        config.allow_gpu_fallback = allow_gpu_fallback;
        config.calibration_cache_path = calibration_cache_path;
        config.tf32 = tf32;

        std::cout << "Building engine with precision=" << yolo::precisionToString(config.precision)
                  << ", workspace_mb=" << workspace_mb << ", tf32=" << (config.tf32 ? "on" : "off")
                  << ", dla_core=" << config.dla_core
                  << ", gpu_fallback=" << (config.allow_gpu_fallback ? "on" : "off") << std::endl;
        if (config.precision == yolo::PrecisionMode::kINT8 && config.calibration_cache_path.empty()) {
            std::cout << "INT8 build will require a calibrator implementation at call site." << std::endl;
        }

        yolo::BuildResult result = yolo::buildEngineFromOnnx(config);
        std::cout << "Build success. Engine size: " << result.engine_size << " bytes" << std::endl;
        for (const auto& tensor : result.tensors) {
            std::cout << (tensor.is_input ? "[input] " : "[output] ") << tensor.name << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "build_engine failed: " << e.what() << std::endl;
        return 1;
    }
}
