#include "BuildEngine.hpp"

#include <NvOnnxParser.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace yolo {
namespace {

class TrtInt8Calibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    explicit TrtInt8Calibrator(Int8Calibrator* delegate) : delegate_(delegate) {}

    int getBatchSize() const noexcept override {
        return delegate_ != nullptr ? delegate_->batchSize() : 0;
    }

    bool getBatch(void* bindings[], const char*[], int) noexcept override {
        if (delegate_ == nullptr || !delegate_->next()) {
            return false;
        }
        bindings[0] = delegate_->nextBatchDevicePointer();
        return bindings[0] != nullptr;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        if (delegate_ == nullptr) {
            length = 0;
            return nullptr;
        }
        return delegate_->readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        if (delegate_ != nullptr) {
            delegate_->writeCalibrationCache(cache, length);
        }
    }

private:
    Int8Calibrator* delegate_{nullptr};
};

}  // namespace

BuildResult buildEngineFromOnnx(const BuildConfig& config, Int8Calibrator* calibrator) {
    BuildResult result;

    if (config.precision == PrecisionMode::kINT8 && calibrator == nullptr && config.calibration_cache_path.empty()) {
        throw std::runtime_error("INT8 build requires a calibrator or calibration cache path");
    }

    TrtLogger logger(config.verbose ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kINFO);

    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder");
    }

    const uint32_t network_flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(network_flags));
    if (!network) {
        throw std::runtime_error("Failed to create TensorRT network");
    }

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        throw std::runtime_error("Failed to create ONNX parser");
    }

    if (!parser->parseFromFile(config.onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::ostringstream oss;
        oss << "Failed to parse ONNX: " << config.onnx_path;
        const int errors = parser->getNbErrors();
        for (int i = 0; i < errors; ++i) {
            oss << "\n - " << parser->getError(i)->desc();
        }
        throw std::runtime_error(oss.str());
    }

    auto profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
        throw std::runtime_error("Failed to create optimization profile");
    }

    nvinfer1::ITensor* input = network->getInput(0);
    if (input == nullptr) {
        throw std::runtime_error("Network has no input tensor");
    }

    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, toDims(config.min_shape));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, toDims(config.opt_shape));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, toDims(config.max_shape));

    auto build_config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!build_config) {
        throw std::runtime_error("Failed to create builder config");
    }

    build_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.workspace_size);
    build_config->addOptimizationProfile(profile);

    if (config.tf32) {
        build_config->setFlag(nvinfer1::BuilderFlag::kTF32);
    } else {
        build_config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }

    if (config.precision == PrecisionMode::kFP16) {
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("FP16 requested but platformHasFastFp16() is false");
        }
        build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    TrtInt8Calibrator trt_calibrator(calibrator);
    if (config.precision == PrecisionMode::kINT8) {
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("INT8 requested but platformHasFastInt8() is false");
        }
        build_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        if (calibrator != nullptr) {
            build_config->setInt8Calibrator(&trt_calibrator);
        }
    }

    if (config.dla_core >= 0) {
        build_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        build_config->setDLACore(config.dla_core);
        if (config.allow_gpu_fallback) {
            build_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
    }

    auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *build_config));
    if (!serialized) {
        throw std::runtime_error("Failed to build serialized TensorRT engine");
    }

    std::ofstream ofs(config.engine_path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open engine output path: " + config.engine_path);
    }
    ofs.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
    ofs.close();

    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime for validation");
    }

    auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine) {
        throw std::runtime_error("Failed to deserialize built TensorRT engine");
    }

    result.success = true;
    result.message = "ok";
    result.engine_size = serialized->size();
    result.tensors = collectTensorInfo(*engine);
    return result;
}

}  // namespace yolo
