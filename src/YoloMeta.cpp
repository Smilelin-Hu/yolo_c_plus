#include "YoloMeta.hpp"

#include <stdexcept>

namespace yolo {
namespace {

ModelMeta makeBaseDetectMeta(const std::string& name) {
    ModelMeta meta;
    meta.model_name = name;
    meta.task = TaskType::kDetection;
    meta.input_width = 640;
    meta.input_height = 640;
    meta.dynamic_shape = true;
    meta.expects_rgb = true;
    meta.normalized_0_1 = true;
    meta.letterbox = true;
    meta.box_format = BoxFormat::kXYWH;
    meta.output_is_decoded_boxes = true;
    meta.conf_threshold = 0.25F;
    meta.iou_threshold = 0.45F;
    return meta;
}

}  // namespace

ModelMeta makeYolo11mMeta(int) {
    auto meta = makeBaseDetectMeta("yolo11m");
    meta.input_tensor_name = "images";
    meta.output_is_decoded_boxes = false;
    meta.apply_nms = true;
    return meta;
}

ModelMeta makeYolo26mMeta(int) {
    auto meta = makeBaseDetectMeta("yolo26m");
    meta.input_tensor_name = "images";
    meta.output_is_decoded_boxes = true;
    meta.apply_nms = false;
    return meta;
}

ModelMeta makeMetaFromPreset(const std::string& preset, int num_classes) {
    if (preset == "yolo11m") {
        return makeYolo11mMeta(num_classes);
    }
    if (preset == "yolo26m") {
        return makeYolo26mMeta(num_classes);
    }
    throw std::runtime_error("Unknown model preset: " + preset);
}

}  // namespace yolo
