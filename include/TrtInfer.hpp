#pragma once

#include "Common.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

struct InferTiming {
    double h2d_ms{0.0};
    double gpu_ms{0.0};
    double d2h_ms{0.0};
    double sync_ms{0.0};
    double total_ms{0.0};
};

struct InferResult {
    std::vector<TensorView> outputs;
    InferTiming timing;
};

struct StageTiming {
    double decode_ms{0.0};
    double warmup_ms{0.0};
    double preprocess_ms{0.0};
    double preprocess_h2d_ms{0.0};
    double preprocess_gpu_ms{0.0};
    double infer_ms{0.0};
    double infer_sync_ms{0.0};
    double postprocess_ms{0.0};
    double postprocess_gpu_ms{0.0};
    double postprocess_sync_ms{0.0};
    double postprocess_d2h_ms{0.0};
    double visualize_ms{0.0};
    double total_ms{0.0};
};

struct TimingSummary {
    std::vector<double> decode_ms;
    std::vector<double> warmup_ms;
    std::vector<double> preprocess_ms;
    std::vector<double> preprocess_h2d_ms;
    std::vector<double> preprocess_gpu_ms;
    std::vector<double> infer_ms;
    std::vector<double> infer_sync_ms;
    std::vector<double> postprocess_ms;
    std::vector<double> postprocess_gpu_ms;
    std::vector<double> postprocess_sync_ms;
    std::vector<double> postprocess_d2h_ms;
    std::vector<double> visualize_ms;
    std::vector<double> total_ms;
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;
    TrtEngine(TrtEngine&&) noexcept;
    TrtEngine& operator=(TrtEngine&&) noexcept;

    const std::vector<TensorInfo>& tensors() const;
    void* runtimeHandle() const;
    void* engineHandle() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class TrtSession {
public:
    explicit TrtSession(const std::string& engine_path, bool use_pinned_output = true);
    ~TrtSession();

    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;

    const std::vector<TensorInfo>& tensors() const;
    void setInputShape(const std::vector<int>& dims);
    void setInputShape(const std::string& tensor_name, const std::vector<int>& dims);
    InferResult infer(const std::vector<float>& input);
    InferResult infer(const float* input, size_t input_bytes);
    InferResult inferFromDevice();
    void* inputDeviceBuffer() const;
    size_t inputDeviceBytes() const;
    cudaStream_t stream() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

double averageMs(const std::vector<double>& values);
void printPerImageTiming(const std::filesystem::path& image_path, const StageTiming& timing, size_t detections,
                         const InferTiming& infer_timing);
void printSummary(size_t processed_images, const TimingSummary& timing);

}  // namespace yolo
