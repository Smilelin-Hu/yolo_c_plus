#pragma once

#include "Common.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

struct BuildConfig {
    std::string onnx_path;
    std::string engine_path;
    ModelMeta meta;
    PrecisionMode precision{PrecisionMode::kFP16};
    size_t workspace_size{1ULL << 30};
    std::vector<int> min_shape{1, 3, 320, 320};
    std::vector<int> opt_shape{1, 3, 640, 640};
    std::vector<int> max_shape{1, 3, 1280, 1280};
    int dla_core{-1};
    bool verbose{false};
    bool allow_gpu_fallback{true};
    std::string calibration_cache_path;
    bool tf32{true};
};

struct BuildResult {
    bool success{false};
    std::string message;
    std::vector<TensorInfo> tensors;
    size_t engine_size{0};
};

class Int8Calibrator {
public:
    virtual ~Int8Calibrator() = default;
    virtual void* nextBatchDevicePointer() = 0;
    virtual int batchSize() const = 0;
    virtual bool next() = 0;
    virtual const void* readCalibrationCache(size_t& length) = 0;
    virtual void writeCalibrationCache(const void* cache, size_t length) = 0;
};

BuildResult buildEngineFromOnnx(const BuildConfig& config, Int8Calibrator* calibrator = nullptr);

}  // namespace yolo
