#pragma once

#include "Types.hpp"

#include <string>

namespace yolo {

ModelMeta makeYolo11mMeta(int num_classes = 80);
ModelMeta makeYolo26mMeta(int num_classes = 80);
ModelMeta makeMetaFromPreset(const std::string& preset, int num_classes = 80);

}  // namespace yolo
