#pragma once

#include "Types.hpp"

#include <filesystem>
#include <vector>

namespace yolo {

bool isSupportedImagePath(const std::filesystem::path& path);
std::vector<std::filesystem::path> collectImagePaths(const std::filesystem::path& input_dir);
std::vector<std::filesystem::path> resolveInputImages(const std::filesystem::path& input_path);

}  // namespace yolo
