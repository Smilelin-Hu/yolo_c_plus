#pragma once

#include "Types.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace yolo {

struct PreprocessTiming {
    double h2d_ms{0.0};
    double gpu_ms{0.0};
    double total_ms{0.0};
};

struct PreprocessResult {
    LetterboxInfo letterbox;
    PreprocessTiming timing;
};

class PreprocessWorkspace {
public:
    explicit PreprocessWorkspace(const RuntimeConfig& config);
    ~PreprocessWorkspace();

    PreprocessWorkspace(const PreprocessWorkspace&) = delete;
    PreprocessWorkspace& operator=(const PreprocessWorkspace&) = delete;

    PreprocessResult run(const cv::Mat& image, const ModelMeta& meta, const RuntimeConfig& config, void* device_input,
                         cudaStream_t stream);

private:
    void ensureSourceBuffer(size_t image_bytes);

    unsigned char* device_source_{nullptr};
    size_t device_source_bytes_{0};
    cudaEvent_t preprocess_start_{nullptr};
    cudaEvent_t preprocess_end_{nullptr};
};

std::vector<float> preprocessImage(const cv::Mat& image, const ModelMeta& meta, const RuntimeConfig& config,
                                   LetterboxInfo& letterbox);

}  // namespace yolo
