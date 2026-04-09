#include "PostprocessDetect.hpp"

#include "Common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace yolo {
namespace {

constexpr int kNmsCandidateMultiplier = 4;

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

    const size_t candidate_limit = std::min(detections.size(), static_cast<size_t>(std::max(max_detections, 1) * kNmsCandidateMultiplier));
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

}  // namespace

std::vector<Detection> decodeDetections(const std::vector<TensorView>& outputs, const ModelMeta& meta,
                                        const RuntimeConfig& config, const LetterboxInfo& letterbox) {
    if (outputs.empty()) {
        return {};
    }

    const TensorView& output = outputs.front();
    if (output.dims.nbDims != 3) {
        throw std::runtime_error("Expected 3D detection output tensor");
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

            detections.push_back(Detection{
                class_id,
                confidence,
                std::clamp((left - pad_x) * inv_scale, 0.0F, max_x),
                std::clamp((top - pad_y) * inv_scale, 0.0F, max_y),
                std::clamp((right - pad_x) * inv_scale, 0.0F, max_x),
                std::clamp((bottom - pad_y) * inv_scale, 0.0F, max_y),
            });
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

            detections.push_back(Detection{
                class_id,
                confidence,
                std::clamp((left - pad_x) * inv_scale, 0.0F, max_x),
                std::clamp((top - pad_y) * inv_scale, 0.0F, max_y),
                std::clamp((right - pad_x) * inv_scale, 0.0F, max_x),
                std::clamp((bottom - pad_y) * inv_scale, 0.0F, max_y),
            });
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

std::vector<Detection> decodeDetections(const std::vector<TensorOutput>& outputs, const ModelMeta& meta,
                                        const RuntimeConfig& config, const LetterboxInfo& letterbox) {
    std::vector<TensorView> views;
    views.reserve(outputs.size());
    for (const auto& output : outputs) {
        views.push_back(TensorView{output.name, output.dims, output.data.data()});
    }
    return decodeDetections(views, meta, config, letterbox);
}

}  // namespace yolo
