#include "VisualizeDetect.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace yolo {
namespace {

cv::Scalar colorForClass(int class_id) {
    static const std::vector<cv::Scalar> palette{
        {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255}, {49, 210, 207},
        {10, 249, 72}, {23, 204, 146}, {134, 219, 61}, {52, 147, 26}, {187, 212, 0},
        {168, 153, 44}, {255, 194, 0}, {147, 69, 52}, {255, 115, 100}, {236, 24, 0},
        {255, 56, 132}, {133, 0, 82}, {255, 56, 203}, {200, 149, 255}, {199, 55, 255},
    };
    const size_t index = static_cast<size_t>(std::max(class_id, 0)) % palette.size();
    return palette[index];
}

}  // namespace

std::string detectionLabel(const Detection& det) {
    std::ostringstream oss;
    oss << det.class_id << ' ' << std::fixed << std::setprecision(2) << det.score;
    return oss.str();
}

void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
    constexpr int thickness = 2;
    constexpr double font_scale = 0.6;
    constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;

    for (const auto& det : detections) {
        const cv::Scalar color = colorForClass(det.class_id);
        const cv::Point top_left(static_cast<int>(std::round(det.left)), static_cast<int>(std::round(det.top)));
        const cv::Point bottom_right(static_cast<int>(std::round(det.right)), static_cast<int>(std::round(det.bottom)));
        cv::rectangle(image, top_left, bottom_right, color, thickness, cv::LINE_AA);

        const std::string label = detectionLabel(det);
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);
        const int box_top = std::max(0, top_left.y - text_size.height - baseline - 6);
        const int box_bottom = box_top + text_size.height + baseline + 6;
        const int box_right = std::min(image.cols - 1, top_left.x + text_size.width + 10);
        cv::rectangle(image, cv::Point(top_left.x, box_top), cv::Point(box_right, box_bottom), color, cv::FILLED);
        cv::putText(image, label, cv::Point(top_left.x + 5, box_bottom - baseline - 3), font_face, font_scale,
                    cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
}

}  // namespace yolo
