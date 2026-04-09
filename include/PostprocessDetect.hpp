#pragma once

#include "Types.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace yolo {

struct PostprocessTiming {
    double gpu_ms{0.0};
    double sync_ms{0.0};
    double d2h_ms{0.0};
    double total_ms{0.0};
};

struct DecodeDetectionsResult {
    std::vector<Detection> detections;
    PostprocessTiming timing;
};

std::vector<Detection> decodeDetections(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                        const RuntimeConfig& config, const LetterboxInfo& letterbox);

std::vector<Detection> decodeDetectionsGpu(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                           const RuntimeConfig& config, const LetterboxInfo& letterbox,
                                           cudaStream_t stream);

DecodeDetectionsResult decodeDetectionsGpuDetailed(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                                   const RuntimeConfig& config, const LetterboxInfo& letterbox,
                                                   cudaStream_t stream);

std::vector<Detection> decodeDetections(const std::vector<TensorOutput>& outputs, const ModelMeta& meta,
                                        const RuntimeConfig& config, const LetterboxInfo& letterbox);

}  // namespace yolo
