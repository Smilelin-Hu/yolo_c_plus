#include "ImageIO.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace yolo {
namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

}  // namespace

bool isSupportedImagePath(const std::filesystem::path& path) {
    if (!path.has_extension()) {
        return false;
    }
    const std::string ext = toLower(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

std::vector<std::filesystem::path> collectImagePaths(const std::filesystem::path& input_dir) {
    std::vector<std::filesystem::path> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::filesystem::path path = entry.path();
        if (isSupportedImagePath(path)) {
            image_paths.push_back(path);
        }
    }
    std::sort(image_paths.begin(), image_paths.end());
    return image_paths;
}

std::vector<std::filesystem::path> resolveInputImages(const std::filesystem::path& input_path) {
    if (std::filesystem::is_directory(input_path)) {
        return collectImagePaths(input_path);
    }
    if (std::filesystem::is_regular_file(input_path) && isSupportedImagePath(input_path)) {
        return {input_path};
    }
    return {};
}

}  // namespace yolo
