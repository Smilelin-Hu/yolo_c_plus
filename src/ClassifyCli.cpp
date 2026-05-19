#include "ClassifyCli.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace yolo {

namespace {

ClassifyScoreMode parseScoreMode(const std::string& value) {
    if (value == "auto") {
        return ClassifyScoreMode::kAuto;
    }
    if (value == "prob" || value == "probabilities") {
        return ClassifyScoreMode::kProbabilities;
    }
    if (value == "logits") {
        return ClassifyScoreMode::kLogits;
    }
    throw std::runtime_error("Unsupported value for --score-mode: " + value +
                             " (expected auto|prob|logits)");
}

}  // namespace

void printClassifyUsage() {
    std::cout << "Usage: classify_infer --engine model.engine --input image_or_dir "
                 "[--output-dir classify_vis] "
                 "[--labels classes.txt] [--imgsz 224] [--input-width 224] [--input-height 224] "
                 "[--topk 5] [--warmup 10] [--score-mode auto|prob|logits] [--benchmark]"
              << std::endl;
}

ClassifyCliConfig parseClassifyCli(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printClassifyUsage();
        throw std::runtime_error("help");
    }
    if (argc < 5) {
        printClassifyUsage();
        throw std::runtime_error("Insufficient arguments");
    }

    ClassifyCliConfig cli;
    cli.runtime.input_width = 0;
    cli.runtime.input_height = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--engine") {
            cli.engine_path = next(arg);
        } else if (arg == "--input") {
            cli.input_path = next(arg);
        } else if (arg == "--output-dir") {
            cli.output_dir_path = next(arg);
        } else if (arg == "--labels") {
            cli.labels_path = next(arg);
        } else if (arg == "--imgsz" || arg == "--input-size") {
            const int size = std::stoi(next(arg));
            cli.runtime.input_width = size;
            cli.runtime.input_height = size;
        } else if (arg == "--input-width") {
            cli.runtime.input_width = std::stoi(next(arg));
        } else if (arg == "--input-height") {
            cli.runtime.input_height = std::stoi(next(arg));
        } else if (arg == "--topk") {
            cli.topk = std::stoi(next(arg));
        } else if (arg == "--warmup") {
            cli.runtime.warmup_runs = std::stoi(next(arg));
        } else if (arg == "--score-mode") {
            cli.score_mode = parseScoreMode(next(arg));
        } else if (arg == "--benchmark") {
            cli.benchmark_mode = true;
        } else if (arg == "--help") {
            printClassifyUsage();
            throw std::runtime_error("help");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cli.engine_path.empty() || cli.input_path.empty()) {
        throw std::runtime_error("--engine and --input are required");
    }
    if (cli.runtime.input_width < 0 || cli.runtime.input_height < 0) {
        throw std::runtime_error("--imgsz/--input-width/--input-height must be non-negative");
    }
    if (cli.topk <= 0) {
        throw std::runtime_error("--topk must be greater than 0");
    }
    if (cli.runtime.warmup_runs < 0) {
        throw std::runtime_error("--warmup must be greater than or equal to 0");
    }

    return cli;
}

}  // namespace yolo
