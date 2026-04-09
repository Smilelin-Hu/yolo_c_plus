#pragma once

#include "Types.hpp"

#include <filesystem>
#include <string>

namespace yolo {

enum class DetectInputKind {
    kImage,
    kDirectory,
};

struct DetectCliConfig {
    std::string engine_path;
    std::string input_path;
    std::string output_dir_path;
    std::string preset{"yolo11m"};
    bool benchmark_mode{false};
    RuntimeConfig runtime;
};

DetectCliConfig parseDetectCli(int argc, char** argv);
void printDetectUsage();
DetectInputKind detectInputKind(const std::filesystem::path& input_path);

}  // namespace yolo
