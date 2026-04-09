#include "TrtInfer.hpp"
#include "DetectCli.hpp"
#include "ImageIO.hpp"
#include "PostprocessDetect.hpp"
#include "Preprocess.hpp"
#include "VisualizeDetect.hpp"
#include "YoloMeta.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void appendTiming(yolo::TimingSummary& summary, const yolo::StageTiming& timing) {
    summary.decode_ms.push_back(timing.decode_ms);
    summary.warmup_ms.push_back(timing.warmup_ms);
    summary.preprocess_ms.push_back(timing.preprocess_ms);
    summary.preprocess_h2d_ms.push_back(timing.preprocess_h2d_ms);
    summary.preprocess_gpu_ms.push_back(timing.preprocess_gpu_ms);
    summary.infer_ms.push_back(timing.infer_ms);
    summary.infer_sync_ms.push_back(timing.infer_sync_ms);
    summary.postprocess_ms.push_back(timing.postprocess_ms);
    summary.postprocess_gpu_ms.push_back(timing.postprocess_gpu_ms);
    summary.postprocess_sync_ms.push_back(timing.postprocess_sync_ms);
    summary.postprocess_d2h_ms.push_back(timing.postprocess_d2h_ms);
    summary.visualize_ms.push_back(timing.visualize_ms);
    summary.total_ms.push_back(timing.total_ms);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        yolo::DetectCliConfig cli = yolo::parseDetectCli(argc, argv);

        const std::filesystem::path input_path(cli.input_path);
        yolo::detectInputKind(input_path);
        const std::vector<std::filesystem::path> image_paths = yolo::resolveInputImages(input_path);
        if (image_paths.empty()) {
            throw std::runtime_error("No supported input images found at: " + cli.input_path);
        }

        const std::filesystem::path output_dir(cli.output_dir_path);
        if (!cli.benchmark_mode) {
            std::filesystem::create_directories(output_dir);
        }

        yolo::ModelMeta meta = yolo::makeMetaFromPreset(cli.preset);
        meta.input_width = cli.runtime.input_width;
        meta.input_height = cli.runtime.input_height;
        meta.conf_threshold = cli.runtime.conf_threshold;
        meta.iou_threshold = cli.runtime.iou_threshold;

        yolo::TrtSession session(cli.engine_path, cli.runtime.use_pinned_output);
        session.setInputShape({1, 3, cli.runtime.input_height, cli.runtime.input_width});
        yolo::PreprocessWorkspace preprocess_workspace(cli.runtime);
        void* device_input = session.inputDeviceBuffer();
        cudaStream_t infer_stream = session.stream();

        bool warmup_done = false;
        yolo::TimingSummary steady_timing_summary;
        size_t processed_images = 0;

        for (const auto& image_path : image_paths) {
            yolo::StageTiming timing;
            const auto total_start = std::chrono::steady_clock::now();

            const auto decode_start = std::chrono::steady_clock::now();
            cv::Mat image = cv::imread(image_path.string());
            const auto decode_end = std::chrono::steady_clock::now();
            if (image.empty()) {
                continue;
            }
            timing.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

            const auto preprocess_start = std::chrono::steady_clock::now();
            yolo::PreprocessResult preprocess =
                preprocess_workspace.run(image, meta, cli.runtime, device_input, infer_stream);
            const auto preprocess_end = std::chrono::steady_clock::now();
            timing.preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
            timing.preprocess_h2d_ms = preprocess.timing.h2d_ms;
            timing.preprocess_gpu_ms = preprocess.timing.gpu_ms;

            if (!warmup_done) {
                const auto warmup_start = std::chrono::steady_clock::now();
                for (int i = 0; i < cli.runtime.warmup_runs; ++i) {
                    session.inferFromDevice();
                }
                const auto warmup_end = std::chrono::steady_clock::now();
                timing.warmup_ms = std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();
                warmup_done = true;
            }

            yolo::InferResult infer_result = session.inferFromDevice();
            timing.infer_ms = infer_result.timing.total_ms;
            timing.infer_sync_ms = infer_result.timing.sync_ms;

            const auto postprocess_start = std::chrono::steady_clock::now();
            yolo::DecodeDetectionsResult decode_result = yolo::decodeDetectionsGpuDetailed(
                infer_result.outputs, meta, cli.runtime, preprocess.letterbox, infer_stream);
            const auto postprocess_end = std::chrono::steady_clock::now();
            timing.postprocess_ms = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
            timing.postprocess_gpu_ms = decode_result.timing.gpu_ms;
            timing.postprocess_sync_ms = decode_result.timing.sync_ms;
            timing.postprocess_d2h_ms = decode_result.timing.d2h_ms;
            std::vector<yolo::Detection> detections = std::move(decode_result.detections);

            const auto visualize_start = std::chrono::steady_clock::now();
            if (!cli.benchmark_mode) {
                cv::Mat annotated = image.clone();
                yolo::drawDetections(annotated, detections);
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
                appendTiming(steady_timing_summary, timing);
            }

            ++processed_images;
            yolo::printPerImageTiming(image_path, timing, detections.size(), infer_result.timing);
        }

        if (processed_images == 0) {
            throw std::runtime_error("No readable images were processed at: " + cli.input_path);
        }

        if (processed_images <= 1) {
            return 0;
        }

        const size_t steady_images = processed_images - 1;
        std::cout << "Steady-state summary (excluding first image):" << std::endl;
        yolo::printSummary(steady_images, steady_timing_summary);
        return 0;
    } catch (const std::exception& e) {
        if (std::string(e.what()) == "help" || std::string(e.what()) == "Insufficient arguments") {
            return std::string(e.what()) == "help" ? 0 : 1;
        }
        std::cerr << "detect_infer failed: " << e.what() << std::endl;
        return 1;
    }
}
