#include "Preprocess.hpp"

#include "Common.hpp"

namespace yolo {
namespace {

__device__ inline float sampleBilinearChannel(const unsigned char* source, int src_width, int src_height, int src_stride,
                                              float x, float y, int channel) {
    const float clamped_x = fminf(fmaxf(x, 0.0F), static_cast<float>(src_width - 1));
    const float clamped_y = fminf(fmaxf(y, 0.0F), static_cast<float>(src_height - 1));

    const int x0 = static_cast<int>(floorf(clamped_x));
    const int y0 = static_cast<int>(floorf(clamped_y));
    const int x1 = min(x0 + 1, src_width - 1);
    const int y1 = min(y0 + 1, src_height - 1);

    const float dx = clamped_x - static_cast<float>(x0);
    const float dy = clamped_y - static_cast<float>(y0);

    const unsigned char* row0 = source + static_cast<size_t>(y0) * static_cast<size_t>(src_stride);
    const unsigned char* row1 = source + static_cast<size_t>(y1) * static_cast<size_t>(src_stride);

    const float v00 = static_cast<float>(row0[x0 * 3 + channel]);
    const float v01 = static_cast<float>(row0[x1 * 3 + channel]);
    const float v10 = static_cast<float>(row1[x0 * 3 + channel]);
    const float v11 = static_cast<float>(row1[x1 * 3 + channel]);

    const float top = v00 + (v01 - v00) * dx;
    const float bottom = v10 + (v11 - v10) * dx;
    return top + (bottom - top) * dy;
}

__global__ void preprocessKernel(const unsigned char* source, int src_width, int src_height, int src_stride,
                                 int dst_width, int dst_height, float scale, float pad_x, float pad_y,
                                 bool expects_rgb, bool normalized_0_1, float* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height) {
        return;
    }

    const size_t dst_area = static_cast<size_t>(dst_width) * static_cast<size_t>(dst_height);
    const size_t index = static_cast<size_t>(y) * static_cast<size_t>(dst_width) + static_cast<size_t>(x);
    float* channel0 = output;
    float* channel1 = output + dst_area;
    float* channel2 = output + dst_area * 2;

    const bool in_image = x >= pad_x && x < (static_cast<float>(dst_width) - pad_x) &&
                          y >= pad_y && y < (static_cast<float>(dst_height) - pad_y);

    const float norm = normalized_0_1 ? (1.0F / 255.0F) : 1.0F;
    if (!in_image) {
        const float pad_value = 114.0F * norm;
        channel0[index] = pad_value;
        channel1[index] = pad_value;
        channel2[index] = pad_value;
        return;
    }

    const float src_x = (static_cast<float>(x) - pad_x + 0.5F) / scale - 0.5F;
    const float src_y = (static_cast<float>(y) - pad_y + 0.5F) / scale - 0.5F;

    const float b = sampleBilinearChannel(source, src_width, src_height, src_stride, src_x, src_y, 0) * norm;
    const float g = sampleBilinearChannel(source, src_width, src_height, src_stride, src_x, src_y, 1) * norm;
    const float r = sampleBilinearChannel(source, src_width, src_height, src_stride, src_x, src_y, 2) * norm;

    if (expects_rgb) {
        channel0[index] = r;
        channel1[index] = g;
        channel2[index] = b;
    } else {
        channel0[index] = b;
        channel1[index] = g;
        channel2[index] = r;
    }
}

}  // namespace

void launchPreprocessKernel(const unsigned char* source, int src_width, int src_height, int src_stride,
                            const ModelMeta& meta, const RuntimeConfig& config, const LetterboxInfo& letterbox,
                            float* device_input, cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((config.input_width + block.x - 1) / block.x, (config.input_height + block.y - 1) / block.y);
    preprocessKernel<<<grid, block, 0, stream>>>(source, src_width, src_height, src_stride, config.input_width,
                                                  config.input_height, letterbox.scale, letterbox.pad_x,
                                                  letterbox.pad_y, meta.expects_rgb, meta.normalized_0_1,
                                                  device_input);
    YOLO_CUDA_CHECK(cudaGetLastError());
}

}  // namespace yolo
