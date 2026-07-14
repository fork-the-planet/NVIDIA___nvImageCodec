// Standalone program that generates 16-bit (uint16 and int16) subsampled JPEG2K
// resource files for Python tests.  Requires nvjpeg2k and CUDA.
//
// Build:
//   nvcc -O2 -o gen_jp2_16bit_subsampled gen_jp2_16bit_subsampled.cpp \
//        -lnvjpeg2k -lcudart -I/usr/include
//
// Run from the repo root:
//   ./gen_jp2_16bit_subsampled <path/to/resources>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg2k.h>

// ---------------------------------------------------------------------------
// Error checkers — throw std::runtime_error on failure
// ---------------------------------------------------------------------------

inline void CheckCudaError(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(code)
            << " (" << code << ") at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}

inline void CheckJ2kError(nvjpeg2kStatus_t code, const char* file, int line)
{
    if (code != NVJPEG2K_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "nvjpeg2k error " << code << " at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}

#define CHECK_CUDA(call) CheckCudaError((call), __FILE__, __LINE__)
#define CHECK_J2K(call)  CheckJ2kError((call), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// RAII helpers
// ---------------------------------------------------------------------------

template<typename T>
struct CudaDeleter { void operator()(T* p) const noexcept { cudaFree(p); } };
template<typename T>
using CudaPtr    = std::unique_ptr<T, CudaDeleter<T>>;
using CudaU8Ptr  = CudaPtr<uint8_t>;
using CudaU16Ptr = CudaPtr<uint16_t>;

// Generic deleter for nvjpeg2k opaque handles (pointer-to-opaque-struct pattern).
// Destroy functions return nvjpeg2kStatus_t (not void), so the template parameter matches that.
template<typename Handle, nvjpeg2kStatus_t(*Destroy)(Handle)>
struct J2kDeleter {
    using pointer = Handle;
    void operator()(Handle h) const noexcept { if (h) Destroy(h); }
};

using J2kEncoderPtr  = std::unique_ptr<std::remove_pointer<nvjpeg2kEncoder_t>::type,
                           J2kDeleter<nvjpeg2kEncoder_t, nvjpeg2kEncoderDestroy>>;
using J2kStatePtr    = std::unique_ptr<std::remove_pointer<nvjpeg2kEncodeState_t>::type,
                           J2kDeleter<nvjpeg2kEncodeState_t, nvjpeg2kEncodeStateDestroy>>;
using J2kParamsPtr   = std::unique_ptr<std::remove_pointer<nvjpeg2kEncodeParams_t>::type,
                           J2kDeleter<nvjpeg2kEncodeParams_t, nvjpeg2kEncodeParamsDestroy>>;

// ---------------------------------------------------------------------------
// fill_uint16 — synthetic planar YCbCr gradient
// ---------------------------------------------------------------------------

// Y plane  (luma_H   x luma_W):   row gradient
// Cb plane (chroma_H x chroma_W): column gradient
// Cr plane (chroma_H x chroma_W): diagonal gradient
// Values in [lo, hi].
static void fill_uint16(
    std::vector<uint16_t>& buf,
    uint32_t luma_H, uint32_t luma_W,
    uint32_t chroma_H, uint32_t chroma_W,
    uint16_t lo, uint16_t hi)
{
    buf.resize(luma_H * luma_W + 2 * chroma_H * chroma_W);
    uint16_t* Y  = buf.data();
    uint16_t* Cb = Y  + luma_H   * luma_W;
    uint16_t* Cr = Cb + chroma_H * chroma_W;

    for (uint32_t r = 0; r < luma_H; ++r)
        for (uint32_t c = 0; c < luma_W; ++c)
            Y[r * luma_W + c] = lo + (uint16_t)((uint32_t)(hi - lo) * r / (luma_H - 1));

    for (uint32_t r = 0; r < chroma_H; ++r)
        for (uint32_t c = 0; c < chroma_W; ++c)
            Cb[r * chroma_W + c] = lo + (uint16_t)((uint32_t)(hi - lo) * c / (chroma_W - 1));

    for (uint32_t r = 0; r < chroma_H; ++r)
        for (uint32_t c = 0; c < chroma_W; ++c)
            Cr[r * chroma_W + c] = lo + (uint16_t)((uint32_t)(hi - lo) * (r + c) / (chroma_H + chroma_W - 2));
}

// ---------------------------------------------------------------------------
// fill_uint8 — synthetic planar YCbCr gradient (uint8 variant of fill_uint16)
// ---------------------------------------------------------------------------
static void fill_uint8(
    std::vector<uint8_t>& buf,
    uint32_t luma_H, uint32_t luma_W,
    uint32_t chroma_H, uint32_t chroma_W,
    uint8_t lo, uint8_t hi)
{
    buf.resize(luma_H * luma_W + 2 * chroma_H * chroma_W);
    uint8_t* Y  = buf.data();
    uint8_t* Cb = Y  + luma_H   * luma_W;
    uint8_t* Cr = Cb + chroma_H * chroma_W;

    for (uint32_t r = 0; r < luma_H; ++r)
        for (uint32_t c = 0; c < luma_W; ++c)
            Y[r * luma_W + c] = lo + (uint8_t)((uint32_t)(hi - lo) * r / (luma_H - 1));

    for (uint32_t r = 0; r < chroma_H; ++r)
        for (uint32_t c = 0; c < chroma_W; ++c)
            Cb[r * chroma_W + c] = lo + (uint8_t)((uint32_t)(hi - lo) * c / (chroma_W - 1));

    for (uint32_t r = 0; r < chroma_H; ++r)
        for (uint32_t c = 0; c < chroma_W; ++c)
            Cr[r * chroma_W + c] = lo + (uint8_t)((uint32_t)(hi - lo) * (r + c) / (chroma_H + chroma_W - 2));
}

// ---------------------------------------------------------------------------
// encode_jp2 — encode one planar YCbCr JPEG2K file (lossy DWT-9/7, Q=90)
// ---------------------------------------------------------------------------
// pixel_data[0] -> Y plane  (luma_H   x luma_W)
// pixel_data[1] -> Cb plane (chroma_H x chroma_W)
// pixel_data[2] -> Cr plane (chroma_H x chroma_W)
// precision: bits actually used (e.g. 8 or 16)
// sgn: 0=unsigned, 1=signed
// pixel_type: NVJPEG2K_UINT8 or NVJPEG2K_UINT16
// Throws std::runtime_error on failure.
template<typename T>
static void encode_jp2(
    nvjpeg2kEncoder_t encoder,
    nvjpeg2kEncodeState_t state,
    const T* host_y,
    const T* host_cb,
    const T* host_cr,
    uint32_t luma_H, uint32_t luma_W,
    uint32_t chroma_H, uint32_t chroma_W,
    uint8_t precision, uint8_t sgn,
    nvjpeg2kImageType_t pixel_type,
    const std::string& out_path)
{
    // Upload planes to GPU — freed automatically on scope exit (including on throw)
    T* raw_y  = nullptr;
    T* raw_cb = nullptr;
    T* raw_cr = nullptr;

    CHECK_CUDA(cudaMalloc(&raw_y,  luma_H   * luma_W   * sizeof(T)));
    CudaPtr<T> d_y(raw_y);
    CHECK_CUDA(cudaMalloc(&raw_cb, chroma_H * chroma_W * sizeof(T)));
    CudaPtr<T> d_cb(raw_cb);
    CHECK_CUDA(cudaMalloc(&raw_cr, chroma_H * chroma_W * sizeof(T)));
    CudaPtr<T> d_cr(raw_cr);

    CHECK_CUDA(cudaMemcpy(d_y.get(),  host_y,  luma_H   * luma_W   * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cb.get(), host_cb, chroma_H * chroma_W * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cr.get(), host_cr, chroma_H * chroma_W * sizeof(T), cudaMemcpyHostToDevice));

    // Component info
    nvjpeg2kImageComponentInfo_t comp_info[3];
    comp_info[0] = {luma_W,   luma_H,   precision, sgn};
    comp_info[1] = {chroma_W, chroma_H, precision, sgn};
    comp_info[2] = {chroma_W, chroma_H, precision, sgn};

    nvjpeg2kEncodeParams_t raw_params = nullptr;
    CHECK_J2K(nvjpeg2kEncodeParamsCreate(&raw_params));
    J2kParamsPtr params(raw_params);

    nvjpeg2kEncodeConfig_t cfg{};
    cfg.stream_type    = NVJPEG2K_STREAM_JP2;
    cfg.color_space    = NVJPEG2K_COLORSPACE_SYCC;
    cfg.image_width    = luma_W;
    cfg.image_height   = luma_H;
    cfg.num_components = 3;
    cfg.image_comp_info = comp_info;
    cfg.code_block_w   = 64;
    cfg.code_block_h   = 64;
    cfg.num_resolutions = 6;
    cfg.prog_order     = NVJPEG2K_PCRL;
    cfg.irreversible   = 1;   // lossy DWT-9/7
    cfg.mct_mode       = 0;   // no MCT; we provide pre-converted YCbCr
    cfg.rsiz           = 0;
    cfg.encode_modes   = 0;

    CHECK_J2K(nvjpeg2kEncodeParamsSetEncodeConfig(params.get(), &cfg));
    CHECK_J2K(nvjpeg2kEncodeParamsSpecifyQuality(params.get(), NVJPEG2K_QUALITY_TYPE_Q_FACTOR, 90.0));

    void* pixel_data[3] = {d_y.get(), d_cb.get(), d_cr.get()};
    size_t pitches[3] = {
        luma_W   * sizeof(T),
        chroma_W * sizeof(T),
        chroma_W * sizeof(T),
    };
    nvjpeg2kImage_t img{};
    img.pixel_data     = pixel_data;
    img.pitch_in_bytes = pitches;
    img.num_components = 3;
    img.pixel_type     = pixel_type;

    CHECK_J2K(nvjpeg2kEncode(encoder, state, params.get(), &img, nullptr));

    // Retrieve bitstream
    size_t length = 0;
    CHECK_J2K(nvjpeg2kEncodeRetrieveBitstream(encoder, state, nullptr, &length, nullptr));
    std::vector<uint8_t> bitstream(length);
    CHECK_J2K(nvjpeg2kEncodeRetrieveBitstream(encoder, state, bitstream.data(), &length, nullptr));

    std::ofstream f(out_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open output file: " + out_path);
    f.write(reinterpret_cast<const char*>(bitstream.data()), (std::streamsize)length);
    f.close();
    std::cout << "Written " << length << " bytes -> " << out_path << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <resources_dir>\n";
        return EXIT_FAILURE;
    }
    std::string res_dir(argv[1]);

    try {
        nvjpeg2kEncoder_t raw_encoder = nullptr;
        nvjpeg2kEncodeState_t raw_state = nullptr;

        CHECK_J2K(nvjpeg2kEncoderCreateSimple(&raw_encoder));
        J2kEncoderPtr encoder(raw_encoder);

        CHECK_J2K(nvjpeg2kEncodeStateCreate(encoder.get(), &raw_state));
        J2kStatePtr state(raw_state);

        const uint32_t H = 512, W = 512;

        const std::string out_dir = res_dir + "/jpeg2k/ycc_rgb_conversion";

        // ---- uint8 (unsigned, sgn=0) ----
        {
            const uint8_t lo = 0, hi = 255;
            std::vector<uint8_t> buf;

            // 4:2:0
            fill_uint8(buf, H, W, H/2, W/2, lo, hi);
            encode_jp2<uint8_t>(encoder.get(), state.get(),
                       buf.data(), buf.data() + H*W, buf.data() + H*W + (H/2)*(W/2),
                       H, W, H/2, W/2, 8, 0, NVJPEG2K_UINT8,
                       out_dir + "/artificial_420_8b3c_uint8.jp2");

            // 4:2:2
            fill_uint8(buf, H, W, H, W/2, lo, hi);
            encode_jp2<uint8_t>(encoder.get(), state.get(),
                       buf.data(), buf.data() + H*W, buf.data() + H*W + H*(W/2),
                       H, W, H, W/2, 8, 0, NVJPEG2K_UINT8,
                       out_dir + "/artificial_422_8b3c_uint8.jp2");

            // 4:4:4
            fill_uint8(buf, H, W, H, W, lo, hi);
            encode_jp2<uint8_t>(encoder.get(), state.get(),
                       buf.data(), buf.data() + H*W, buf.data() + 2*H*W,
                       H, W, H, W, 8, 0, NVJPEG2K_UINT8,
                       out_dir + "/artificial_444_8b3c_uint8.jp2");
        }

        // ---- uint16 (unsigned, sgn=0) ----
        // Use [4096, 61440] so gradient avoids edges; neutral chroma at 32768.
        {
            const uint16_t lo = 4096, hi = 61440;
            std::vector<uint16_t> buf;

            // 4:2:0
            fill_uint16(buf, H, W, H/2, W/2, lo, hi);
            encode_jp2<uint16_t>(encoder.get(), state.get(),
                       buf.data(), buf.data() + H*W, buf.data() + H*W + (H/2)*(W/2),
                       H, W, H/2, W/2, 16, 0, NVJPEG2K_UINT16,
                       out_dir + "/artificial_420_16b3c_uint16.jp2");

            // 4:2:2
            fill_uint16(buf, H, W, H, W/2, lo, hi);
            encode_jp2<uint16_t>(encoder.get(), state.get(),
                       buf.data(), buf.data() + H*W, buf.data() + H*W + H*(W/2),
                       H, W, H, W/2, 16, 0, NVJPEG2K_UINT16,
                       out_dir + "/artificial_422_16b3c_uint16.jp2");

            // 4:4:4
            fill_uint16(buf, H, W, H, W, lo, hi);
            encode_jp2<uint16_t>(encoder.get(), state.get(),
                       buf.data(), buf.data() + H*W, buf.data() + 2*H*W,
                       H, W, H, W, 16, 0, NVJPEG2K_UINT16,
                       out_dir + "/artificial_444_16b3c_uint16.jp2");
        }

        // ---- int16 (signed, sgn=1) ----
        // Same gradient but shifted: [lo-32768, hi-32768] -> [-28672, 28672].
        // Store as two's-complement bit pattern in uint16 buffer.
        {
            const uint16_t lo = 4096, hi = 61440;
            std::vector<uint16_t> buf_u;
            std::vector<uint16_t> buf_s;

            // 4:2:0
            fill_uint16(buf_u, H, W, H/2, W/2, lo, hi);
            buf_s.resize(buf_u.size());
            for (size_t i = 0; i < buf_u.size(); ++i)
                buf_s[i] = (uint16_t)((int16_t)(buf_u[i] - 32768));

            encode_jp2<uint16_t>(encoder.get(), state.get(),
                       buf_s.data(), buf_s.data() + H*W, buf_s.data() + H*W + (H/2)*(W/2),
                       H, W, H/2, W/2, 16, 1, NVJPEG2K_UINT16,
                       out_dir + "/artificial_420_16b3c_int16.jp2");

            // 4:2:2
            fill_uint16(buf_u, H, W, H, W/2, lo, hi);
            buf_s.resize(buf_u.size());
            for (size_t i = 0; i < buf_u.size(); ++i)
                buf_s[i] = (uint16_t)((int16_t)(buf_u[i] - 32768));

            encode_jp2<uint16_t>(encoder.get(), state.get(),
                       buf_s.data(), buf_s.data() + H*W, buf_s.data() + H*W + H*(W/2),
                       H, W, H, W/2, 16, 1, NVJPEG2K_UINT16,
                       out_dir + "/artificial_422_16b3c_int16.jp2");

            // 4:4:4
            fill_uint16(buf_u, H, W, H, W, lo, hi);
            buf_s.resize(buf_u.size());
            for (size_t i = 0; i < buf_u.size(); ++i)
                buf_s[i] = (uint16_t)((int16_t)(buf_u[i] - 32768));

            encode_jp2<uint16_t>(encoder.get(), state.get(),
                       buf_s.data(), buf_s.data() + H*W, buf_s.data() + 2*H*W,
                       H, W, H, W, 16, 1, NVJPEG2K_UINT16,
                       out_dir + "/artificial_444_16b3c_int16.jp2");
        }


        std::cout << "Done.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
