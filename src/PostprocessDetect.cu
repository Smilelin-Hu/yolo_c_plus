#include "PostprocessDetect.hpp"

#include "Common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace yolo {
namespace {

constexpr int kNmsCandidateMultiplier = 4;
constexpr int kThreadsPerBlock = 256;
constexpr int kDecodedBoxFields = 6;

float clampValue(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

struct GpuDetectionCandidate {
    int class_id;
    float score;
    float left;
    float top;
    float right;
    float bottom;
};

bool isDecodedBoxesOutput(const TensorView& output, const ModelMeta& meta) {
    if (!meta.output_is_decoded_boxes || output.dims.nbDims != 3) {
        return false;
    }
    return output.dims.d[2] == kDecodedBoxFields;
}

Detection makeDetectionFromDecodedBox(const float* pred, const LetterboxInfo& letterbox) {
    const float inv_scale = 1.0F / letterbox.scale;
    const float max_x = static_cast<float>(letterbox.image_width - 1);
    const float max_y = static_cast<float>(letterbox.image_height - 1);
    return Detection{static_cast<int>(pred[5]),
                     pred[4],
                     clampValue((pred[0] - letterbox.pad_x) * inv_scale, 0.0F, max_x),
                     clampValue((pred[1] - letterbox.pad_y) * inv_scale, 0.0F, max_y),
                     clampValue((pred[2] - letterbox.pad_x) * inv_scale, 0.0F, max_x),
                     clampValue((pred[3] - letterbox.pad_y) * inv_scale, 0.0F, max_y)};
}

std::vector<Detection> decodeDecodedBoxesCpu(const TensorView& output, const RuntimeConfig& config,
                                             const LetterboxInfo& letterbox) {
    const size_t num_preds = static_cast<size_t>(output.dims.d[1] > 0 ? output.dims.d[1] : 0);
    std::vector<Detection> detections;
    detections.reserve(std::min(num_preds, static_cast<size_t>(std::max(config.max_detections, 1))));

    const float* data = output.data;
    for (size_t i = 0; i < num_preds; ++i) {
        const float* pred = data + i * kDecodedBoxFields;
        if (pred[4] < config.conf_threshold) {
            continue;
        }
        detections.push_back(makeDetectionFromDecodedBox(pred, letterbox));
        if (detections.size() >= static_cast<size_t>(config.max_detections)) {
            break;
        }
    }
    return detections;
}

__device__ float deviceIou(const GpuDetectionCandidate& a, const GpuDetectionCandidate& b) {
    const float left = max(a.left, b.left);
    const float top = max(a.top, b.top);
    const float right = min(a.right, b.right);
    const float bottom = min(a.bottom, b.bottom);
    const float width = max(0.0F, right - left);
    const float height = max(0.0F, bottom - top);
    const float inter = width * height;
    const float area_a = max(0.0F, a.right - a.left) * max(0.0F, a.bottom - a.top);
    const float area_b = max(0.0F, b.right - b.left) * max(0.0F, b.bottom - b.top);
    return inter / max(area_a + area_b - inter, 1e-6F);
}

__global__ void decodeDetectionsKernel(const float* data, OutputLayout layout, int channels, size_t num_preds,
                                       bool has_objectness, BoxFormat box_format, float conf_threshold,
                                       float inv_scale, float pad_x, float pad_y, float max_x, float max_y,
                                       GpuDetectionCandidate* candidates, unsigned int* candidate_count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_preds) {
        return;
    }

    const size_t class_offset = has_objectness ? 5U : 4U;
    const int num_classes = channels - 4 - (has_objectness ? 1 : 0);
    float objectness = 1.0F;
    float class_score = 0.0F;
    int class_id = -1;
    float left = 0.0F;
    float top = 0.0F;
    float right = 0.0F;
    float bottom = 0.0F;

    if (layout == OutputLayout::kBCN) {
        const float* x_ptr = data;
        const float* y_ptr = data + num_preds;
        const float* w_ptr = data + num_preds * 2;
        const float* h_ptr = data + num_preds * 3;
        const float* objectness_ptr = has_objectness ? data + num_preds * 4 : nullptr;
        const float* class_ptr = data + num_preds * class_offset;

        objectness = objectness_ptr != nullptr ? objectness_ptr[index] : 1.0F;
        const float* class_scores = class_ptr + index;
        for (int c = 0; c < num_classes; ++c) {
            const float score = class_scores[static_cast<size_t>(c) * num_preds];
            if (score > class_score) {
                class_score = score;
                class_id = c;
            }
        }

        left = x_ptr[index];
        top = y_ptr[index];
        right = w_ptr[index];
        bottom = h_ptr[index];
        if (box_format == BoxFormat::kXYWH) {
            const float half_w = w_ptr[index] * 0.5F;
            const float half_h = h_ptr[index] * 0.5F;
            left = x_ptr[index] - half_w;
            top = y_ptr[index] - half_h;
            right = x_ptr[index] + half_w;
            bottom = y_ptr[index] + half_h;
        }
    } else {
        const size_t pred_stride = static_cast<size_t>(channels);
        const float* pred = data + index * pred_stride;
        objectness = has_objectness ? pred[4] : 1.0F;
        for (int c = 0; c < num_classes; ++c) {
            const float score = pred[class_offset + static_cast<size_t>(c)];
            if (score > class_score) {
                class_score = score;
                class_id = c;
            }
        }

        left = pred[0];
        top = pred[1];
        right = pred[2];
        bottom = pred[3];
        if (box_format == BoxFormat::kXYWH) {
            const float half_w = pred[2] * 0.5F;
            const float half_h = pred[3] * 0.5F;
            left = pred[0] - half_w;
            top = pred[1] - half_h;
            right = pred[0] + half_w;
            bottom = pred[1] + half_h;
        }
    }

    const float confidence = objectness * class_score;
    if (confidence < conf_threshold || class_id < 0) {
        return;
    }

    const unsigned int candidate_index = atomicAdd(candidate_count, 1U);
    GpuDetectionCandidate candidate;
    candidate.class_id = class_id;
    candidate.score = confidence;
    candidate.left = min(max((left - pad_x) * inv_scale, 0.0F), max_x);
    candidate.top = min(max((top - pad_y) * inv_scale, 0.0F), max_y);
    candidate.right = min(max((right - pad_x) * inv_scale, 0.0F), max_x);
    candidate.bottom = min(max((bottom - pad_y) * inv_scale, 0.0F), max_y);
    candidates[candidate_index] = candidate;
}

float intersectionOverUnion(const Detection& a, const Detection& b) {
    const float left = std::max(a.left, b.left);
    const float top = std::max(a.top, b.top);
    const float right = std::min(a.right, b.right);
    const float bottom = std::min(a.bottom, b.bottom);
    const float width = std::max(0.0F, right - left);
    const float height = std::max(0.0F, bottom - top);
    const float inter = width * height;
    const float area_a = std::max(0.0F, a.right - a.left) * std::max(0.0F, a.bottom - a.top);
    const float area_b = std::max(0.0F, b.right - b.left) * std::max(0.0F, b.bottom - b.top);
    return inter / std::max(area_a + area_b - inter, 1e-6F);
}

std::vector<Detection> runNms(std::vector<Detection> detections, float iou_threshold, int max_detections) {
    std::sort(detections.begin(), detections.end(), [](const Detection& lhs, const Detection& rhs) {
        return lhs.score > rhs.score;
    });

    const size_t candidate_limit =
        std::min(detections.size(), static_cast<size_t>(std::max(max_detections, 1) * kNmsCandidateMultiplier));
    detections.resize(candidate_limit);

    std::vector<Detection> kept;
    kept.reserve(std::min(candidate_limit, static_cast<size_t>(std::max(max_detections, 1))));
    std::vector<uint8_t> removed(candidate_limit, 0);

    for (size_t i = 0; i < candidate_limit; ++i) {
        if (removed[i] != 0) {
            continue;
        }
        const Detection& current = detections[i];
        kept.push_back(current);
        if (static_cast<int>(kept.size()) >= max_detections) {
            break;
        }
        for (size_t j = i + 1; j < candidate_limit; ++j) {
            if (removed[j] != 0 || current.class_id != detections[j].class_id) {
                continue;
            }
            if (intersectionOverUnion(current, detections[j]) > iou_threshold) {
                removed[j] = 1;
            }
        }
    }
    return kept;
}

std::vector<Detection> decodeDetectionsCpu(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                           const RuntimeConfig& config, const LetterboxInfo& letterbox) {
    if (outputs.empty()) {
        return {};
    }

    const TensorView& output = outputs.front();
    if (output.dims.nbDims != 3) {
        throw std::runtime_error("Expected 3D detection output tensor");
    }
    if (isDecodedBoxesOutput(output, meta)) {
        return decodeDecodedBoxesCpu(output, config, letterbox);
    }

    const int feature_dim = output.dims.d[1];
    const int pred_dim = output.dims.d[2];
    const int alt_feature_dim = output.dims.d[2];
    const int alt_pred_dim = output.dims.d[1];

    OutputLayout layout = meta.output_layout;
    int channels = 0;
    size_t num_preds = 0;
    if (feature_dim > 4 && pred_dim > 0) {
        layout = OutputLayout::kBCN;
        channels = feature_dim;
        num_preds = static_cast<size_t>(pred_dim);
    } else if (alt_feature_dim > 4 && alt_pred_dim > 0) {
        layout = OutputLayout::kBNC;
        channels = alt_feature_dim;
        num_preds = static_cast<size_t>(alt_pred_dim);
    } else {
        throw std::runtime_error("Unable to infer detection output layout from shape " + dimsToString(output.dims));
    }

    const int num_classes = channels - 4 - (meta.has_objectness ? 1 : 0);
    if (num_classes <= 0) {
        throw std::runtime_error("Unable to infer class count from output shape " + dimsToString(output.dims));
    }

    const float conf_threshold = config.conf_threshold;
    const float inv_scale = 1.0F / letterbox.scale;
    const float pad_x = letterbox.pad_x;
    const float pad_y = letterbox.pad_y;
    const float max_x = static_cast<float>(letterbox.image_width - 1);
    const float max_y = static_cast<float>(letterbox.image_height - 1);
    const size_t class_offset = meta.has_objectness ? 5U : 4U;
    const size_t score_offset = meta.has_objectness ? 4U : 0U;
    const float* data = output.data;

    std::vector<Detection> detections;
    detections.reserve(std::min(num_preds, static_cast<size_t>(config.max_detections * 8)));

    if (layout == OutputLayout::kBCN) {
        const float* x_ptr = data;
        const float* y_ptr = data + num_preds;
        const float* w_ptr = data + num_preds * 2;
        const float* h_ptr = data + num_preds * 3;
        const float* objectness_ptr = meta.has_objectness ? data + num_preds * score_offset : nullptr;
        const float* class_ptr = data + num_preds * class_offset;

        for (size_t i = 0; i < num_preds; ++i) {
            const float objectness = objectness_ptr != nullptr ? objectness_ptr[i] : 1.0F;
            float class_score = 0.0F;
            int class_id = -1;
            const float* class_scores = class_ptr + i;
            for (int c = 0; c < num_classes; ++c) {
                const float score = class_scores[static_cast<size_t>(c) * num_preds];
                if (score > class_score) {
                    class_score = score;
                    class_id = c;
                }
            }

            const float confidence = objectness * class_score;
            if (confidence < conf_threshold) {
                continue;
            }

            float left = x_ptr[i];
            float top = y_ptr[i];
            float right = w_ptr[i];
            float bottom = h_ptr[i];
            if (meta.box_format == BoxFormat::kXYWH) {
                const float half_w = w_ptr[i] * 0.5F;
                const float half_h = h_ptr[i] * 0.5F;
                left = x_ptr[i] - half_w;
                top = y_ptr[i] - half_h;
                right = x_ptr[i] + half_w;
                bottom = y_ptr[i] + half_h;
            }

            detections.push_back(Detection{class_id,
                                           confidence,
                                           clampValue((left - pad_x) * inv_scale, 0.0F, max_x),
                                           clampValue((top - pad_y) * inv_scale, 0.0F, max_y),
                                           clampValue((right - pad_x) * inv_scale, 0.0F, max_x),
                                           clampValue((bottom - pad_y) * inv_scale, 0.0F, max_y)});
        }
    } else {
        const size_t pred_stride = static_cast<size_t>(channels);
        for (size_t i = 0; i < num_preds; ++i) {
            const float* pred = data + i * pred_stride;
            const float objectness = meta.has_objectness ? pred[4] : 1.0F;
            float class_score = 0.0F;
            int class_id = -1;
            for (int c = 0; c < num_classes; ++c) {
                const float score = pred[class_offset + static_cast<size_t>(c)];
                if (score > class_score) {
                    class_score = score;
                    class_id = c;
                }
            }

            const float confidence = objectness * class_score;
            if (confidence < conf_threshold) {
                continue;
            }

            float left = pred[0];
            float top = pred[1];
            float right = pred[2];
            float bottom = pred[3];
            if (meta.box_format == BoxFormat::kXYWH) {
                const float half_w = pred[2] * 0.5F;
                const float half_h = pred[3] * 0.5F;
                left = pred[0] - half_w;
                top = pred[1] - half_h;
                right = pred[0] + half_w;
                bottom = pred[1] + half_h;
            }

            detections.push_back(Detection{class_id,
                                           confidence,
                                           clampValue((left - pad_x) * inv_scale, 0.0F, max_x),
                                           clampValue((top - pad_y) * inv_scale, 0.0F, max_y),
                                           clampValue((right - pad_x) * inv_scale, 0.0F, max_x),
                                           clampValue((bottom - pad_y) * inv_scale, 0.0F, max_y)});
        }
    }

    if (meta.apply_nms) {
        return runNms(std::move(detections), config.iou_threshold, config.max_detections);
    }

    std::sort(detections.begin(), detections.end(), [](const Detection& lhs, const Detection& rhs) {
        return lhs.score > rhs.score;
    });
    if (detections.size() > static_cast<size_t>(config.max_detections)) {
        detections.resize(static_cast<size_t>(config.max_detections));
    }

    return detections;
}

double elapsedMs(const std::chrono::steady_clock::time_point& start,
                 const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

std::vector<Detection> decodeDetections(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                        const RuntimeConfig& config, const LetterboxInfo& letterbox) {
    return decodeDetectionsCpu(outputs, meta, config, letterbox);
}

DecodeDetectionsResult decodeDetectionsGpuDetailed(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                                   const RuntimeConfig& config, const LetterboxInfo& letterbox,
                                                   cudaStream_t stream) {
    DecodeDetectionsResult result;
    const auto total_start = std::chrono::steady_clock::now();
    if (outputs.empty()) {
        result.timing.total_ms = 0.0;
        return result;
    }

    const TensorView& output = outputs.front();
    if (output.dims.nbDims != 3) {
        result.detections = decodeDetectionsCpu(outputs, meta, config, letterbox);
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
        return result;
    }
    if (isDecodedBoxesOutput(output, meta)) {
        result.detections = decodeDecodedBoxesCpu(output, config, letterbox);
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
        return result;
    }
    if (output.device_data == nullptr || output.data == nullptr) {
        result.detections = decodeDetectionsCpu(outputs, meta, config, letterbox);
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
        return result;
    }

    const int feature_dim = output.dims.d[1];
    const int pred_dim = output.dims.d[2];
    const int alt_feature_dim = output.dims.d[2];
    const int alt_pred_dim = output.dims.d[1];

    OutputLayout layout = meta.output_layout;
    int channels = 0;
    size_t num_preds = 0;
    if (feature_dim > 4 && pred_dim > 0) {
        layout = OutputLayout::kBCN;
        channels = feature_dim;
        num_preds = static_cast<size_t>(pred_dim);
    } else if (alt_feature_dim > 4 && alt_pred_dim > 0) {
        layout = OutputLayout::kBNC;
        channels = alt_feature_dim;
        num_preds = static_cast<size_t>(alt_pred_dim);
    } else {
        result.detections = decodeDetectionsCpu(outputs, meta, config, letterbox);
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
        return result;
    }

    const int num_classes = channels - 4 - (meta.has_objectness ? 1 : 0);
    if (num_classes <= 0) {
        result.detections = decodeDetectionsCpu(outputs, meta, config, letterbox);
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
        return result;
    }

    const size_t max_candidates = num_preds;
    if (max_candidates == 0) {
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
        return result;
    }

    GpuDetectionCandidate* device_candidates = nullptr;
    unsigned int* device_candidate_count = nullptr;
    std::vector<GpuDetectionCandidate> host_candidates(max_candidates);
    unsigned int host_candidate_count = 0;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_end = nullptr;
    YOLO_CUDA_CHECK(cudaEventCreate(&gpu_start));
    YOLO_CUDA_CHECK(cudaEventCreate(&gpu_end));

    YOLO_CUDA_CHECK(cudaMalloc(&device_candidates, max_candidates * sizeof(GpuDetectionCandidate)));
    YOLO_CUDA_CHECK(cudaMalloc(&device_candidate_count, sizeof(unsigned int)));
    YOLO_CUDA_CHECK(cudaMemsetAsync(device_candidate_count, 0, sizeof(unsigned int), stream));

    const float inv_scale = 1.0F / letterbox.scale;
    const float max_x = static_cast<float>(letterbox.image_width - 1);
    const float max_y = static_cast<float>(letterbox.image_height - 1);
    const int blocks = static_cast<int>((num_preds + kThreadsPerBlock - 1) / kThreadsPerBlock);
    YOLO_CUDA_CHECK(cudaEventRecord(gpu_start, stream));
    decodeDetectionsKernel<<<blocks, kThreadsPerBlock>>>(output.device_data, layout, channels, num_preds,
                                                         meta.has_objectness, meta.box_format,
                                                         config.conf_threshold, inv_scale, letterbox.pad_x,
                                                         letterbox.pad_y, max_x, max_y, device_candidates,
                                                         device_candidate_count);
    YOLO_CUDA_CHECK(cudaGetLastError());
    YOLO_CUDA_CHECK(cudaEventRecord(gpu_end, stream));

    const auto sync_start = std::chrono::steady_clock::now();
    YOLO_CUDA_CHECK(cudaMemcpyAsync(&host_candidate_count, device_candidate_count, sizeof(unsigned int),
                                    cudaMemcpyDeviceToHost, stream));
    YOLO_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto sync_end = std::chrono::steady_clock::now();

    float gpu_ms = 0.0F;
    YOLO_CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end));
    result.timing.gpu_ms = static_cast<double>(gpu_ms);
    result.timing.sync_ms = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

    host_candidate_count = std::min<unsigned int>(host_candidate_count, static_cast<unsigned int>(max_candidates));
    if (host_candidate_count == 0) {
        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_end);
        cudaFree(device_candidate_count);
        cudaFree(device_candidates);
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
        return result;
    }

    const auto d2h_start = std::chrono::steady_clock::now();
    YOLO_CUDA_CHECK(cudaMemcpy(host_candidates.data(), device_candidates,
                               static_cast<size_t>(host_candidate_count) * sizeof(GpuDetectionCandidate),
                               cudaMemcpyDeviceToHost));
    const auto d2h_end = std::chrono::steady_clock::now();
    result.timing.d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    cudaFree(device_candidate_count);
    cudaFree(device_candidates);

    result.detections.reserve(host_candidate_count);
    for (unsigned int i = 0; i < host_candidate_count; ++i) {
        const auto& candidate = host_candidates[i];
        result.detections.push_back(Detection{candidate.class_id, candidate.score, candidate.left, candidate.top,
                                              candidate.right, candidate.bottom});
    }

    if (meta.apply_nms) {
        result.detections = runNms(std::move(result.detections), config.iou_threshold, config.max_detections);
    } else {
        std::sort(result.detections.begin(), result.detections.end(), [](const Detection& lhs, const Detection& rhs) {
            return lhs.score > rhs.score;
        });
        if (result.detections.size() > static_cast<size_t>(config.max_detections)) {
            result.detections.resize(static_cast<size_t>(config.max_detections));
        }
    }

    result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
    return result;
}

std::vector<Detection> decodeDetectionsGpu(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                           const RuntimeConfig& config, const LetterboxInfo& letterbox,
                                           cudaStream_t stream) {
    return decodeDetectionsGpuDetailed(outputs, meta, config, letterbox, stream).detections;
}

std::vector<Detection> decodeDetections(const std::vector<TensorOutput>& outputs, const ModelMeta& meta,
                                        const RuntimeConfig& config, const LetterboxInfo& letterbox) {
    std::vector<TensorView> views;
    views.reserve(outputs.size());
    for (const auto& output : outputs) {
        views.push_back(TensorView{output.name, output.dims, output.data.data(), nullptr});
    }
    return decodeDetections(views, meta, config, letterbox);
}

}  // namespace yolo
