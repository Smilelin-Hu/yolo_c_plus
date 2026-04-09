#include "TrtInfer.hpp"

#include <NvInferRuntime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace yolo {
namespace {

struct BufferBlock {
    void* ptr{nullptr};
    size_t bytes{0};
};

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

size_t byteSize(const nvinfer1::Dims& dims, nvinfer1::DataType type) {
    return static_cast<size_t>(volume(dims)) * elementSize(type);
}

}  // namespace

class TrtEngine::Impl {
public:
    explicit Impl(const std::string& engine_path) : logger_(nvinfer1::ILogger::Severity::kINFO) {
        blob_ = readBinaryFile(engine_path);
        runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(blob_.data(), blob_.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
        tensors_ = collectTensorInfo(*engine_);
    }

    TrtLogger logger_;
    std::vector<char> blob_;
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    std::vector<TensorInfo> tensors_;
};

TrtEngine::TrtEngine(const std::string& engine_path) : impl_(std::make_unique<Impl>(engine_path)) {}
TrtEngine::~TrtEngine() = default;
TrtEngine::TrtEngine(TrtEngine&&) noexcept = default;
TrtEngine& TrtEngine::operator=(TrtEngine&&) noexcept = default;
const std::vector<TensorInfo>& TrtEngine::tensors() const { return impl_->tensors_; }
void* TrtEngine::runtimeHandle() const { return impl_->runtime_.get(); }
void* TrtEngine::engineHandle() const { return impl_->engine_.get(); }

class TrtSession::Impl {
public:
    explicit Impl(const std::string& engine_path, bool use_pinned_output)
        : engine_(engine_path), use_pinned_output_(use_pinned_output) {
        auto* cuda_engine = static_cast<nvinfer1::ICudaEngine*>(engine_.engineHandle());
        context_ = TrtUniquePtr<nvinfer1::IExecutionContext>(cuda_engine->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context");
        }
        YOLO_CUDA_CHECK(cudaStreamCreate(&stream_));
        YOLO_CUDA_CHECK(cudaEventCreate(&infer_start_));
        YOLO_CUDA_CHECK(cudaEventCreate(&infer_end_));
        prepareBindings();
    }

    ~Impl() {
        for (auto& entry : device_buffers_) {
            if (entry.second.ptr != nullptr) {
                cudaFree(entry.second.ptr);
            }
        }
        for (auto& entry : host_buffers_) {
            if (entry.second.ptr != nullptr) {
                if (use_pinned_output_) {
                    cudaFreeHost(entry.second.ptr);
                } else {
                    std::free(entry.second.ptr);
                }
            }
        }
        if (infer_start_ != nullptr) {
            cudaEventDestroy(infer_start_);
        }
        if (infer_end_ != nullptr) {
            cudaEventDestroy(infer_end_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    void reserveDeviceBuffer(const std::string& tensor_name, size_t bytes) {
        BufferBlock& block = device_buffers_[tensor_name];
        if (block.bytes >= bytes) {
            return;
        }
        if (block.ptr != nullptr) {
            cudaFree(block.ptr);
        }
        void* device_ptr = nullptr;
        YOLO_CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
        block.ptr = device_ptr;
        block.bytes = bytes;
        context_->setTensorAddress(tensor_name.c_str(), device_ptr);
    }

    void reserveHostBuffer(const std::string& tensor_name, size_t bytes) {
        BufferBlock& block = host_buffers_[tensor_name];
        if (block.bytes >= bytes) {
            return;
        }
        if (block.ptr != nullptr) {
            if (use_pinned_output_) {
                cudaFreeHost(block.ptr);
            } else {
                std::free(block.ptr);
            }
        }
        void* host_ptr = nullptr;
        if (use_pinned_output_) {
            YOLO_CUDA_CHECK(cudaMallocHost(&host_ptr, bytes));
        } else {
            host_ptr = std::malloc(bytes);
            if (host_ptr == nullptr) {
                throw std::bad_alloc();
            }
        }
        block.ptr = host_ptr;
        block.bytes = bytes;
    }

    void prepareBindings() {
        for (const auto& tensor : engine_.tensors()) {
            context_->setTensorAddress(tensor.name.c_str(), nullptr);
            if (!tensor.is_input) {
                host_buffers_[tensor.name] = BufferBlock{};
            }
            device_buffers_[tensor.name] = BufferBlock{};
        }
    }

    void ensureBuffersForCurrentShapes() {
        for (const auto& tensor : engine_.tensors()) {
            const nvinfer1::Dims actual_dims = context_->getTensorShape(tensor.name.c_str());
            const size_t bytes = std::max<size_t>(byteSize(actual_dims, tensor.dtype), 1);
            reserveDeviceBuffer(tensor.name, bytes);
            if (!tensor.is_input) {
                reserveHostBuffer(tensor.name, bytes);
            }
        }
    }

    InferResult infer(const std::vector<float>& input) {
        return infer(input.data(), input.size() * sizeof(float));
    }

    InferResult infer(const float* input, size_t input_bytes) {
        const auto& tensors = engine_.tensors();
        auto input_it = std::find_if(tensors.begin(), tensors.end(), [](const TensorInfo& info) { return info.is_input; });
        if (input_it == tensors.end()) {
            throw std::runtime_error("No input tensor found in engine");
        }

        ensureBuffersForCurrentShapes();

        const nvinfer1::Dims input_dims = context_->getTensorShape(input_it->name.c_str());
        const size_t expected_input_bytes = byteSize(input_dims, input_it->dtype);
        if (input_bytes != expected_input_bytes) {
            throw std::runtime_error("Input size mismatch for tensor " + input_it->name + ": got " +
                                     std::to_string(input_bytes) + " bytes, expected " +
                                     std::to_string(expected_input_bytes) + " bytes");
        }

        const auto h2d_start = std::chrono::steady_clock::now();
        YOLO_CUDA_CHECK(cudaMemcpyAsync(device_buffers_.at(input_it->name).ptr, input, input_bytes,
                                        cudaMemcpyHostToDevice, stream_));
        const auto h2d_end = std::chrono::steady_clock::now();

        InferResult result = inferPreparedInput();
        result.timing.h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
        result.timing.total_ms += result.timing.h2d_ms;
        return result;
    }

    InferResult inferPreparedInput() {
        const auto total_start = std::chrono::steady_clock::now();
        const auto& tensors = engine_.tensors();

        InferResult result;

        YOLO_CUDA_CHECK(cudaEventRecord(infer_start_, stream_));
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
        YOLO_CUDA_CHECK(cudaEventRecord(infer_end_, stream_));

        const auto d2h_start = std::chrono::steady_clock::now();
        result.outputs.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            if (tensor.is_input) {
                continue;
            }
            TensorView output;
            output.name = tensor.name;
            output.dims = context_->getTensorShape(tensor.name.c_str());
            const size_t bytes = std::max<size_t>(byteSize(output.dims, tensor.dtype), 1);
            auto* host_ptr = static_cast<float*>(host_buffers_.at(tensor.name).ptr);
            YOLO_CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_buffers_.at(tensor.name).ptr, bytes,
                                            cudaMemcpyDeviceToHost, stream_));
            output.data = host_ptr;
            output.device_data = static_cast<const float*>(device_buffers_.at(tensor.name).ptr);
            result.outputs.push_back(std::move(output));
        }
        const auto d2h_end = std::chrono::steady_clock::now();

        const auto sync_start = std::chrono::steady_clock::now();
        YOLO_CUDA_CHECK(cudaEventSynchronize(infer_end_));
        float gpu_ms = 0.0F;
        YOLO_CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, infer_start_, infer_end_));
        YOLO_CUDA_CHECK(cudaStreamSynchronize(stream_));
        const auto sync_end = std::chrono::steady_clock::now();
        const auto total_end = std::chrono::steady_clock::now();

        result.timing.h2d_ms = 0.0;
        result.timing.gpu_ms = static_cast<double>(gpu_ms);
        result.timing.d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
        result.timing.sync_ms = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();
        result.timing.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        return result;
    }

    void setInputShape(const std::vector<int>& dims) {
        const auto& tensors = engine_.tensors();
        auto input_it = std::find_if(tensors.begin(), tensors.end(), [](const TensorInfo& info) { return info.is_input; });
        if (input_it == tensors.end()) {
            throw std::runtime_error("No input tensor found in engine");
        }
        setInputShape(input_it->name, dims);
    }

    void setInputShape(const std::string& tensor_name, const std::vector<int>& dims) {
        if (!context_->setInputShape(tensor_name.c_str(), toDims(dims))) {
            throw std::runtime_error("Failed to set TensorRT input shape for " + tensor_name);
        }
        ensureBuffersForCurrentShapes();
    }

    const std::vector<TensorInfo>& tensors() const { return engine_.tensors(); }

    void* inputDeviceBuffer() const {
        const auto& tensors = engine_.tensors();
        auto input_it = std::find_if(tensors.begin(), tensors.end(), [](const TensorInfo& info) { return info.is_input; });
        if (input_it == tensors.end()) {
            return nullptr;
        }
        auto buffer_it = device_buffers_.find(input_it->name);
        return buffer_it == device_buffers_.end() ? nullptr : buffer_it->second.ptr;
    }

    size_t inputDeviceBytes() const {
        const auto& tensors = engine_.tensors();
        auto input_it = std::find_if(tensors.begin(), tensors.end(), [](const TensorInfo& info) { return info.is_input; });
        if (input_it == tensors.end()) {
            return 0;
        }
        const nvinfer1::Dims input_dims = context_->getTensorShape(input_it->name.c_str());
        return byteSize(input_dims, input_it->dtype);
    }

    cudaStream_t stream() const { return stream_; }

    TrtEngine engine_;
    bool use_pinned_output_{true};
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{nullptr};
    cudaEvent_t infer_start_{nullptr};
    cudaEvent_t infer_end_{nullptr};
    std::unordered_map<std::string, BufferBlock> device_buffers_;
    std::unordered_map<std::string, BufferBlock> host_buffers_;
};

TrtSession::TrtSession(const std::string& engine_path, bool use_pinned_output)
    : impl_(std::make_unique<Impl>(engine_path, use_pinned_output)) {}
TrtSession::~TrtSession() = default;
const std::vector<TensorInfo>& TrtSession::tensors() const { return impl_->tensors(); }
void TrtSession::setInputShape(const std::vector<int>& dims) { impl_->setInputShape(dims); }
void TrtSession::setInputShape(const std::string& tensor_name, const std::vector<int>& dims) {
    impl_->setInputShape(tensor_name, dims);
}
InferResult TrtSession::infer(const std::vector<float>& input) { return impl_->infer(input); }
InferResult TrtSession::infer(const float* input, size_t input_bytes) { return impl_->infer(input, input_bytes); }
InferResult TrtSession::inferFromDevice() { return impl_->inferPreparedInput(); }
void* TrtSession::inputDeviceBuffer() const { return impl_->inputDeviceBuffer(); }
size_t TrtSession::inputDeviceBytes() const { return impl_->inputDeviceBytes(); }
cudaStream_t TrtSession::stream() const { return impl_->stream(); }

double averageMs(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    return total / static_cast<double>(values.size());
}

void printPerImageTiming(const std::filesystem::path& image_path, const StageTiming& timing, size_t detections,
                         const InferTiming& infer_timing) {
    std::cout << image_path.filename().string() << " decode=" << std::fixed << std::setprecision(2)
              << timing.decode_ms << " ms warmup=" << timing.warmup_ms << " ms preprocess=" << timing.preprocess_ms
              << " ms(h2d=" << timing.preprocess_h2d_ms << ", gpu=" << timing.preprocess_gpu_ms << ") infer_total="
              << timing.infer_ms << " ms gpu=" << infer_timing.gpu_ms << " ms h2d=" << infer_timing.h2d_ms
              << " ms d2h=" << infer_timing.d2h_ms << " ms sync=" << infer_timing.sync_ms << " ms post="
              << timing.postprocess_ms << " ms(gpu=" << timing.postprocess_gpu_ms << ", sync="
              << timing.postprocess_sync_ms << ", d2h=" << timing.postprocess_d2h_ms << ") vis="
              << timing.visualize_ms << " ms total=" << timing.total_ms << " ms detections=" << detections
              << std::endl;
}

void printSummary(size_t processed_images, const TimingSummary& timing) {
    std::cout << "Processed images: " << processed_images << std::endl;
    std::cout << "Average decode latency: " << std::fixed << std::setprecision(2) << averageMs(timing.decode_ms)
              << " ms" << std::endl;
    std::cout << "Average warmup latency: " << std::fixed << std::setprecision(2) << averageMs(timing.warmup_ms)
              << " ms" << std::endl;
    std::cout << "Average preprocess latency: " << std::fixed << std::setprecision(2)
              << averageMs(timing.preprocess_ms) << " ms" << std::endl;
    std::cout << "  preprocess h2d: " << std::fixed << std::setprecision(2) << averageMs(timing.preprocess_h2d_ms)
              << " ms" << std::endl;
    std::cout << "  preprocess gpu: " << std::fixed << std::setprecision(2) << averageMs(timing.preprocess_gpu_ms)
              << " ms" << std::endl;
    std::cout << "Average inference latency: " << std::fixed << std::setprecision(2) << averageMs(timing.infer_ms)
              << " ms" << std::endl;
    std::cout << "  inference sync: " << std::fixed << std::setprecision(2) << averageMs(timing.infer_sync_ms)
              << " ms" << std::endl;
    std::cout << "Average postprocess latency: " << std::fixed << std::setprecision(2)
              << averageMs(timing.postprocess_ms) << " ms" << std::endl;
    std::cout << "  postprocess gpu: " << std::fixed << std::setprecision(2) << averageMs(timing.postprocess_gpu_ms)
              << " ms" << std::endl;
    std::cout << "  postprocess sync: " << std::fixed << std::setprecision(2)
              << averageMs(timing.postprocess_sync_ms) << " ms" << std::endl;
    std::cout << "  postprocess d2h: " << std::fixed << std::setprecision(2) << averageMs(timing.postprocess_d2h_ms)
              << " ms" << std::endl;
    std::cout << "Average visualize latency: " << std::fixed << std::setprecision(2)
              << averageMs(timing.visualize_ms) << " ms" << std::endl;
    std::cout << "Average total latency: " << std::fixed << std::setprecision(2) << averageMs(timing.total_ms)
              << " ms" << std::endl;
}

}  // namespace yolo
