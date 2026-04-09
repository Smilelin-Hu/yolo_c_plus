#include "Preprocess.hpp"

#include "Common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace yolo {

void launchPreprocessKernel(const unsigned char* source, int src_width, int src_height, int src_stride,
                            const ModelMeta& meta, const RuntimeConfig& config, const LetterboxInfo& letterbox,
                            float* device_input, cudaStream_t stream);

namespace {

void fillLetterboxInfo(const cv::Mat& image, const ModelMeta& meta, const RuntimeConfig& config, LetterboxInfo& letterbox) {
    letterbox.image_width = image.cols;
    letterbox.image_height = image.rows;
    letterbox.input_width = config.input_width;
    letterbox.input_height = config.input_height;

    const float scale_x = static_cast<float>(config.input_width) / static_cast<float>(image.cols);
    const float scale_y = static_cast<float>(config.input_height) / static_cast<float>(image.rows);
    letterbox.scale = meta.letterbox ? std::min(scale_x, scale_y) : scale_x;
}

}  // namespace

PreprocessWorkspace::PreprocessWorkspace(const RuntimeConfig&) {
    YOLO_CUDA_CHECK(cudaEventCreate(&preprocess_start_));
    YOLO_CUDA_CHECK(cudaEventCreate(&preprocess_end_));
}

PreprocessWorkspace::~PreprocessWorkspace() {
    if (device_source_ != nullptr) {
        cudaFree(device_source_);
    }
    if (preprocess_start_ != nullptr) {
        cudaEventDestroy(preprocess_start_);
    }
    if (preprocess_end_ != nullptr) {
        cudaEventDestroy(preprocess_end_);
    }
}

void PreprocessWorkspace::ensureSourceBuffer(size_t image_bytes) {
    if (device_source_bytes_ >= image_bytes) {
        return;
    }
    if (device_source_ != nullptr) {
        cudaFree(device_source_);
    }
    YOLO_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_source_), image_bytes));
    device_source_bytes_ = image_bytes;
}

PreprocessResult PreprocessWorkspace::run(const cv::Mat& image, const ModelMeta& meta, const RuntimeConfig& config,
                                          void* device_input, cudaStream_t stream) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    if (image.type() != CV_8UC3) {
        throw std::runtime_error("Only CV_8UC3 input images are supported for GPU preprocessing");
    }
    if (device_input == nullptr) {
        throw std::runtime_error("TensorRT input device buffer is null");
    }

    PreprocessResult result;
    fillLetterboxInfo(image, meta, config, result.letterbox);

    const int resized_w = std::max(1, static_cast<int>(std::round(image.cols * result.letterbox.scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(image.rows * result.letterbox.scale)));
    result.letterbox.pad_x = static_cast<float>(config.input_width - resized_w) * 0.5F;
    result.letterbox.pad_y = static_cast<float>(config.input_height - resized_h) * 0.5F;

    const size_t image_bytes = image.step[0] * static_cast<size_t>(image.rows);
    ensureSourceBuffer(image_bytes);

    const auto total_start = std::chrono::steady_clock::now();
    const auto h2d_start = std::chrono::steady_clock::now();
    YOLO_CUDA_CHECK(cudaMemcpyAsync(device_source_, image.data, image_bytes, cudaMemcpyHostToDevice, stream));
    const auto h2d_end = std::chrono::steady_clock::now();

    YOLO_CUDA_CHECK(cudaEventRecord(preprocess_start_, stream));
    launchPreprocessKernel(device_source_, image.cols, image.rows, static_cast<int>(image.step[0]), meta, config,
                           result.letterbox, static_cast<float*>(device_input), stream);
    YOLO_CUDA_CHECK(cudaEventRecord(preprocess_end_, stream));
    YOLO_CUDA_CHECK(cudaEventSynchronize(preprocess_end_));
    const auto total_end = std::chrono::steady_clock::now();

    float gpu_ms = 0.0F;
    YOLO_CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, preprocess_start_, preprocess_end_));
    result.timing.h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    result.timing.gpu_ms = static_cast<double>(gpu_ms);
    result.timing.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    return result;
}

std::vector<float> preprocessImage(const cv::Mat&, const ModelMeta&, const RuntimeConfig&, LetterboxInfo&) {
    throw std::runtime_error("CPU preprocessImage is not supported in the GPU preprocessing path");
}

}  // namespace yolo
