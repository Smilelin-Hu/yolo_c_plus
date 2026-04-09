#pragma once

#include <NvInfer.h>

#include <array>
#include <string>
#include <vector>

namespace yolo {

enum class TaskType {
    kDetection,
    kClassification,
    kSegmentation,
};

enum class PrecisionMode {
    kFP32,
    kFP16,
    kINT8,
};

enum class OutputLayout {
    kBNC,
    kBCN,
};

enum class BoxFormat {
    kXYWH,
    kXYXY,
};

struct TensorInfo {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    bool is_input{false};
};

struct TensorView {
    std::string name;
    nvinfer1::Dims dims{};
    const float* data{nullptr};
    const float* device_data{nullptr};
};

struct TensorOutput {
    std::string name;
    nvinfer1::Dims dims{};
    std::vector<float> data;
};

struct LetterboxInfo {
    float scale{1.0F};
    float pad_x{0.0F};
    float pad_y{0.0F};
    int input_width{0};
    int input_height{0};
    int image_width{0};
    int image_height{0};
};

struct Detection {
    int class_id{0};
    float score{0.0F};
    float left{0.0F};
    float top{0.0F};
    float right{0.0F};
    float bottom{0.0F};
};

struct ModelMeta {
    std::string model_name;
    TaskType task{TaskType::kDetection};
    int num_classes{80};
    std::vector<std::string> class_names;
    int input_width{640};
    int input_height{640};
    bool dynamic_shape{true};
    bool expects_rgb{true};
    bool normalized_0_1{true};
    bool letterbox{true};
    bool apply_nms{true};
    std::string input_tensor_name;
    std::vector<std::string> output_tensor_names;
    OutputLayout output_layout{OutputLayout::kBNC};
    bool has_objectness{false};
    BoxFormat box_format{BoxFormat::kXYWH};
    bool output_is_decoded_boxes{true};
    float conf_threshold{0.25F};
    float iou_threshold{0.45F};
};

struct RuntimeConfig {
    int device_id{0};
    int input_width{640};
    int input_height{640};
    float conf_threshold{0.25F};
    float iou_threshold{0.45F};
    int warmup_runs{10};
    int max_detections{300};
    bool use_pinned_output{true};
    bool use_pinned_input{true};
};

}  // namespace yolo
