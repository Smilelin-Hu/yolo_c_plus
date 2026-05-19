#include "ClassifyCli.hpp"
#include "ImageIO.hpp"
#include "TrtInfer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ClassifyTiming {
    double decode_ms{0.0};
    double preprocess_ms{0.0};
    double infer_ms{0.0};
    double visualize_ms{0.0};
    double total_ms{0.0};
};

struct ClassifyTimingSummary {
    std::vector<double> decode_ms;
    std::vector<double> preprocess_ms;
    std::vector<double> infer_ms;
    std::vector<double> visualize_ms;
    std::vector<double> total_ms;
};

struct ClassPrediction {
    int class_id{0};
    float score{0.0F};
};

struct ScoreView {
    std::vector<float> values;
    bool used_softmax{false};
};

const char* scoreModeDescription(yolo::ClassifyScoreMode mode) {
    switch (mode) {
        case yolo::ClassifyScoreMode::kAuto:
            return "auto";
        case yolo::ClassifyScoreMode::kProbabilities:
            return "probabilities";
        case yolo::ClassifyScoreMode::kLogits:
            return "logits";
    }
    return "auto";
}

void appendTiming(ClassifyTimingSummary& summary, const ClassifyTiming& timing) {
    summary.decode_ms.push_back(timing.decode_ms);
    summary.preprocess_ms.push_back(timing.preprocess_ms);
    summary.infer_ms.push_back(timing.infer_ms);
    summary.visualize_ms.push_back(timing.visualize_ms);
    summary.total_ms.push_back(timing.total_ms);
}

double averageMs(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    return total / static_cast<double>(values.size());
}

std::vector<std::string> loadClassNames(const std::string& labels_path) {
    std::ifstream ifs(labels_path);
    if (!ifs) {
        throw std::runtime_error("Failed to open labels file: " + labels_path);
    }

    std::vector<std::string> names;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

std::vector<std::string> makeDefaultClassNames(size_t count) {
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        names.push_back(std::to_string(i));
    }
    return names;
}

std::vector<std::string> resolveClassNames(const std::string& labels_path, size_t class_count) {
    if (labels_path.empty()) {
        return makeDefaultClassNames(class_count);
    }

    std::vector<std::string> names = loadClassNames(labels_path);
    if (names.size() != class_count) {
        throw std::runtime_error("Labels file class count mismatch: got " + std::to_string(names.size()) +
                                 ", expected " + std::to_string(class_count));
    }
    return names;
}

std::pair<int, int> resolveInputSize(const yolo::ClassifyCliConfig& cli, const yolo::TrtSession& session) {
    int input_width = cli.runtime.input_width;
    int input_height = cli.runtime.input_height;

    if (input_width > 0 && input_height == 0) {
        input_height = input_width;
    }
    if (input_height > 0 && input_width == 0) {
        input_width = input_height;
    }

    const auto& tensors = session.tensors();
    auto input_it = std::find_if(tensors.begin(), tensors.end(), [](const yolo::TensorInfo& tensor) {
        return tensor.is_input;
    });
    if (input_it == tensors.end()) {
        throw std::runtime_error("No input tensor found in engine");
    }

    if ((input_width <= 0 || input_height <= 0) && input_it->dims.nbDims >= 4) {
        if (input_it->dims.d[2] > 0) {
            input_height = input_it->dims.d[2];
        }
        if (input_it->dims.d[3] > 0) {
            input_width = input_it->dims.d[3];
        }
    }

    if (input_width <= 0 || input_height <= 0) {
        input_width = 224;
        input_height = 224;
    }
    return {input_width, input_height};
}

cv::Mat resizeForClassification(const cv::Mat& image, int input_width, int input_height) {
    cv::Mat resized;
    if (input_width == input_height) {
        const int shorter_side = std::min(image.cols, image.rows);
        if (shorter_side <= 0) {
            throw std::runtime_error("Invalid image size");
        }
        const double scale = static_cast<double>(input_width) / static_cast<double>(shorter_side);
        const int resized_width = std::max(1, static_cast<int>(std::lround(image.cols * scale)));
        const int resized_height = std::max(1, static_cast<int>(std::lround(image.rows * scale)));
        cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        cv::resize(image, resized, cv::Size(input_width, input_height), 0.0, 0.0, cv::INTER_LINEAR);
    }
    return resized;
}

cv::Mat centerCrop(const cv::Mat& image, int input_width, int input_height) {
    if (image.cols < input_width || image.rows < input_height) {
        throw std::runtime_error("Image is smaller than requested center crop");
    }

    const int left = (image.cols - input_width) / 2;
    const int top = (image.rows - input_height) / 2;
    return image(cv::Rect(left, top, input_width, input_height)).clone();
}

std::vector<float> preprocessClassificationImage(const cv::Mat& image, int input_width, int input_height) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    if (image.type() != CV_8UC3) {
        throw std::runtime_error("Only CV_8UC3 input images are supported");
    }

    const cv::Mat resized = resizeForClassification(image, input_width, input_height);
    const cv::Mat cropped = centerCrop(resized, input_width, input_height);

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input(static_cast<size_t>(3 * input_width * input_height));
    const size_t plane = static_cast<size_t>(input_width * input_height);
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            const cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
            const size_t offset = static_cast<size_t>(y * input_width + x);
            input[offset] = static_cast<float>(pixel[0]) / 255.0F;
            input[plane + offset] = static_cast<float>(pixel[1]) / 255.0F;
            input[2 * plane + offset] = static_cast<float>(pixel[2]) / 255.0F;
        }
    }
    return input;
}

size_t inferClassCount(const yolo::TensorView& output) {
    const int64_t total_values = yolo::volume(output.dims);
    if (total_values <= 0) {
        throw std::runtime_error("Classification output tensor is empty");
    }
    if (output.dims.nbDims > 1 && output.dims.d[0] > 1) {
        throw std::runtime_error("Only batch size 1 classification inference is supported");
    }
    return static_cast<size_t>(total_values);
}

bool looksLikeProbabilities(const yolo::TensorView& output, size_t class_count) {
    double sum = 0.0;
    for (size_t i = 0; i < class_count; ++i) {
        const float value = output.data[i];
        if (value < 0.0F || value > 1.0F) {
            return false;
        }
        sum += value;
    }
    return std::abs(sum - 1.0) <= 1e-3;
}

ScoreView makeDisplayScores(const yolo::TensorView& output, yolo::ClassifyScoreMode score_mode) {
    const size_t class_count = inferClassCount(output);
    ScoreView scores;
    scores.values.resize(class_count);

    if (score_mode == yolo::ClassifyScoreMode::kProbabilities ||
        (score_mode == yolo::ClassifyScoreMode::kAuto && looksLikeProbabilities(output, class_count))) {
        for (size_t i = 0; i < class_count; ++i) {
            scores.values[i] = output.data[i];
        }
        return scores;
    }

    const float max_logit = *std::max_element(output.data, output.data + class_count);
    double sum = 0.0;
    for (size_t i = 0; i < class_count; ++i) {
        const float value = std::exp(output.data[i] - max_logit);
        scores.values[i] = value;
        sum += value;
    }
    if (sum > 0.0) {
        for (float& value : scores.values) {
            value = static_cast<float>(value / sum);
        }
    }
    scores.used_softmax = true;
    return scores;
}

std::vector<ClassPrediction> topKPredictions(const std::vector<float>& scores, int topk) {
    const size_t class_count = scores.size();
    const size_t k = std::min(class_count, static_cast<size_t>(topk));
    std::vector<int> indices(class_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k), indices.end(),
                      [&](int lhs, int rhs) { return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)]; });

    std::vector<ClassPrediction> predictions;
    predictions.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const int class_id = indices[i];
        predictions.push_back(ClassPrediction{class_id, scores[static_cast<size_t>(class_id)]});
    }
    return predictions;
}

std::string formatPredictions(const std::vector<ClassPrediction>& predictions,
                              const std::vector<std::string>& class_names) {
    std::ostringstream oss;
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& pred = predictions[i];
        if (i != 0) {
            oss << ", ";
        }
        oss << class_names[static_cast<size_t>(pred.class_id)] << ' ' << std::fixed << std::setprecision(4)
            << pred.score;
    }
    return oss.str();
}

void drawClassificationPredictions(cv::Mat& image, const std::vector<ClassPrediction>& predictions,
                                   const std::vector<std::string>& class_names) {
    if (image.empty() || predictions.empty()) {
        return;
    }

    constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double font_scale = 0.7;
    constexpr int thickness = 2;
    constexpr int padding = 10;
    constexpr int line_gap = 8;

    std::vector<std::string> lines;
    lines.reserve(predictions.size());
    int max_width = 0;
    int total_height = padding;
    std::vector<cv::Size> text_sizes;
    text_sizes.reserve(predictions.size());

    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& pred = predictions[i];
        std::ostringstream oss;
        oss << (i + 1) << ". " << class_names[static_cast<size_t>(pred.class_id)] << " "
            << std::fixed << std::setprecision(2) << pred.score * 100.0F << '%';
        lines.push_back(oss.str());

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(lines.back(), font_face, font_scale, thickness, &baseline);
        text_sizes.push_back(text_size);
        max_width = std::max(max_width, text_size.width);
        total_height += text_size.height + baseline + line_gap;
    }
    total_height += padding - line_gap;

    const int box_width = std::min(image.cols, max_width + padding * 2);
    const int box_height = std::min(image.rows, total_height);
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(box_width, box_height), cv::Scalar(20, 20, 20), cv::FILLED);
    cv::addWeighted(overlay, 0.6, image, 0.4, 0.0, image);

    int y = padding;
    for (size_t i = 0; i < lines.size(); ++i) {
        int baseline = 0;
        cv::getTextSize(lines[i], font_face, font_scale, thickness, &baseline);
        y += text_sizes[i].height;
        cv::putText(image, lines[i], cv::Point(padding, y), font_face, font_scale, cv::Scalar(255, 255, 255),
                    thickness, cv::LINE_AA);
        y += baseline + line_gap;
    }
}

void printSummary(size_t processed_images, const ClassifyTimingSummary& summary) {
    std::cout << "Processed images: " << processed_images << std::endl;
    std::cout << "Average decode latency: " << std::fixed << std::setprecision(2)
              << averageMs(summary.decode_ms) << " ms" << std::endl;
    std::cout << "Average preprocess latency: " << std::fixed << std::setprecision(2)
              << averageMs(summary.preprocess_ms) << " ms" << std::endl;
    std::cout << "Average inference latency: " << std::fixed << std::setprecision(2)
              << averageMs(summary.infer_ms) << " ms" << std::endl;
    std::cout << "Average visualize latency: " << std::fixed << std::setprecision(2)
              << averageMs(summary.visualize_ms) << " ms" << std::endl;
    std::cout << "Average total latency: " << std::fixed << std::setprecision(2)
              << averageMs(summary.total_ms) << " ms" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        yolo::ClassifyCliConfig cli = yolo::parseClassifyCli(argc, argv);

        const std::filesystem::path input_path(cli.input_path);
        const std::vector<std::filesystem::path> image_paths = yolo::resolveInputImages(input_path);
        if (image_paths.empty()) {
            throw std::runtime_error("No supported input images found at: " + cli.input_path);
        }
        const std::filesystem::path output_dir(cli.output_dir_path);
        if (!cli.benchmark_mode) {
            std::filesystem::create_directories(output_dir);
        }

        yolo::TrtSession session(cli.engine_path, cli.runtime.use_pinned_output);
        const auto [input_width, input_height] = resolveInputSize(cli, session);
        session.setInputShape({1, 3, input_height, input_width});

        bool warmup_done = false;
        bool class_names_ready = false;
        bool score_mode_reported = false;
        std::vector<std::string> class_names;
        ClassifyTimingSummary steady_summary;
        size_t processed_images = 0;

        for (const auto& image_path : image_paths) {
            ClassifyTiming timing;
            const auto total_start = std::chrono::steady_clock::now();

            const auto decode_start = std::chrono::steady_clock::now();
            cv::Mat image = cv::imread(image_path.string());
            const auto decode_end = std::chrono::steady_clock::now();
            if (image.empty()) {
                continue;
            }
            timing.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

            const auto preprocess_start = std::chrono::steady_clock::now();
            const std::vector<float> input = preprocessClassificationImage(image, input_width, input_height);
            const auto preprocess_end = std::chrono::steady_clock::now();
            timing.preprocess_ms =
                std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

            if (!warmup_done && cli.runtime.warmup_runs > 0) {
                for (int i = 0; i < cli.runtime.warmup_runs; ++i) {
                    session.infer(input);
                }
                warmup_done = true;
            } else if (!warmup_done) {
                warmup_done = true;
            }

            yolo::InferResult infer_result = session.infer(input);
            timing.infer_ms = infer_result.timing.total_ms;

            if (infer_result.outputs.empty()) {
                throw std::runtime_error("Classification engine produced no output tensors");
            }
            const yolo::TensorView& output = infer_result.outputs.front();
            if (output.data == nullptr) {
                throw std::runtime_error("Classification output host buffer is null");
            }

            if (!class_names_ready) {
                class_names = resolveClassNames(cli.labels_path, inferClassCount(output));
                class_names_ready = true;
            }

            const ScoreView display_scores = makeDisplayScores(output, cli.score_mode);
            if (!score_mode_reported) {
                if (display_scores.used_softmax) {
                    std::cout << "Score mode=" << scoreModeDescription(cli.score_mode)
                              << "; model output is treated as logits and displayed as softmax probabilities."
                              << std::endl;
                } else if (cli.score_mode == yolo::ClassifyScoreMode::kProbabilities) {
                    std::cout << "Score mode=probabilities; output is displayed directly without softmax."
                              << std::endl;
                }
                score_mode_reported = true;
            }
            const std::vector<ClassPrediction> predictions = topKPredictions(display_scores.values, cli.topk);

            const auto visualize_start = std::chrono::steady_clock::now();
            if (!cli.benchmark_mode) {
                cv::Mat annotated = image.clone();
                drawClassificationPredictions(annotated, predictions, class_names);
                const std::filesystem::path output_path = output_dir / image_path.filename();
                if (!cv::imwrite(output_path.string(), annotated)) {
                    throw std::runtime_error("Failed to write output image: " + output_path.string());
                }
            }
            const auto visualize_end = std::chrono::steady_clock::now();
            timing.visualize_ms = std::chrono::duration<double, std::milli>(visualize_end - visualize_start).count();

            const auto total_end = std::chrono::steady_clock::now();
            timing.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

            if (processed_images > 0) {
                appendTiming(steady_summary, timing);
            }

            ++processed_images;
            const ClassPrediction& top1 = predictions.front();
            std::cout << image_path.filename().string() << " top1="
                      << class_names[static_cast<size_t>(top1.class_id)] << ' ' << std::fixed
                      << std::setprecision(4) << top1.score << " top"
                      << std::min(static_cast<size_t>(cli.topk), predictions.size()) << "=["
                      << formatPredictions(predictions, class_names) << "] decode=" << std::setprecision(2)
                      << timing.decode_ms << " ms preprocess=" << timing.preprocess_ms << " ms infer="
                      << timing.infer_ms << " ms vis=" << timing.visualize_ms << " ms total=" << timing.total_ms
                      << " ms" << std::endl;
        }

        if (processed_images == 0) {
            throw std::runtime_error("No readable images were processed at: " + cli.input_path);
        }

        if (processed_images > 1) {
            std::cout << "Steady-state summary (excluding first image):" << std::endl;
            printSummary(processed_images - 1, steady_summary);
        }
        return 0;
    } catch (const std::exception& e) {
        if (std::string(e.what()) == "help" || std::string(e.what()) == "Insufficient arguments") {
            return std::string(e.what()) == "help" ? 0 : 1;
        }
        std::cerr << "classify_infer failed: " << e.what() << std::endl;
        return 1;
    }
}
