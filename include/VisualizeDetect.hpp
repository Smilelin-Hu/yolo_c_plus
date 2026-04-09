#pragma once

#include "Types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace yolo {

std::string detectionLabel(const Detection& det);
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);

}  // namespace yolo
