#pragma once

#include "Types.hpp"

#include <string>

namespace yolo {

enum class ClassifyScoreMode {
    kAuto,
    kProbabilities,
    kLogits,
};

struct ClassifyCliConfig {
    std::string engine_path;
    std::string input_path;
    std::string output_dir_path{"classify_vis"};
    std::string labels_path;
    int topk{5};
    bool benchmark_mode{false};
    ClassifyScoreMode score_mode{ClassifyScoreMode::kAuto};
    RuntimeConfig runtime;
};

ClassifyCliConfig parseClassifyCli(int argc, char** argv);
void printClassifyUsage();

}  // namespace yolo
