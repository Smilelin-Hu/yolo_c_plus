#include "DetectCli.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace yolo {

void printDetectUsage() {
    std::cout << "Usage: detect_infer --engine model.engine --input image_or_dir --output-dir vis "
                 "[--preset yolo11m|yolo26m] [--input-width 640] [--input-height 640] "
                 "[--conf 0.25] [--iou 0.45] [--warmup 10] [--benchmark]"
              << std::endl;
}

DetectInputKind detectInputKind(const std::filesystem::path& input_path) {
    if (!std::filesystem::exists(input_path)) {
        throw std::runtime_error("Input path does not exist: " + input_path.string());
    }
    if (std::filesystem::is_directory(input_path)) {
        return DetectInputKind::kDirectory;
    }
    if (std::filesystem::is_regular_file(input_path)) {
        return DetectInputKind::kImage;
    }
    throw std::runtime_error("Unsupported input path: " + input_path.string());
}

DetectCliConfig parseDetectCli(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printDetectUsage();
        throw std::runtime_error("help");
    }
    if (argc < 7) {
        printDetectUsage();
        throw std::runtime_error("Insufficient arguments");
    }

    DetectCliConfig cli;
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
        } else if (arg == "--preset") {
            cli.preset = next(arg);
        } else if (arg == "--input-width") {
            cli.runtime.input_width = std::stoi(next(arg));
        } else if (arg == "--input-height") {
            cli.runtime.input_height = std::stoi(next(arg));
        } else if (arg == "--conf") {
            cli.runtime.conf_threshold = std::stof(next(arg));
        } else if (arg == "--iou") {
            cli.runtime.iou_threshold = std::stof(next(arg));
        } else if (arg == "--warmup") {
            cli.runtime.warmup_runs = std::stoi(next(arg));
        } else if (arg == "--benchmark") {
            cli.benchmark_mode = true;
        } else if (arg == "--help") {
            printDetectUsage();
            throw std::runtime_error("help");
        }
    }

    if (cli.engine_path.empty() || cli.input_path.empty() || cli.output_dir_path.empty()) {
        throw std::runtime_error("--engine, --input and --output-dir are required");
    }

    return cli;
}

}  // namespace yolo
