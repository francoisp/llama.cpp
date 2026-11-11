#include "prefix-mask.cuh"
#include "convert.cuh"

#define CUDA_PREFIX_MASK_BLOCK_SIZE 256

// res[c, i] = 0 when c < n_visible[i], else -inf
template <typename T>
static __global__ void prefix_mask_kernel(
        const int32_t * __restrict__ n_visible, T * __restrict__ dst,
        const int64_t n_kv, const int64_t n_rows, const int64_t ne1, const int64_t ne2,
        const int64_t s1, const int64_t s2, const int64_t s3) {
    const T zero = ggml_cuda_cast<T>(0.0f);
    const T ninf = ggml_cuda_cast<T>(-INFINITY);

    for (int64_t i = blockIdx.y; i < n_rows; i += gridDim.y) {
        const int64_t i1 = i % ne1;
        const int64_t i2 = (i / ne1) % ne2;
        const int64_t i3 = i / (ne1*ne2);

        T * row = dst + i1*s1 + i2*s2 + i3*s3;

        const int64_t n = n_visible[i];
        for (int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; c < n_kv; c += (int64_t) gridDim.x*blockDim.x) {
            row[c] = c < n ? zero : ninf;
        }
    }
}

template <typename T>
static void prefix_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * n_visible = dst->src[0];

    GGML_ASSERT(n_visible->type == GGML_TYPE_I32 && ggml_is_contiguous(n_visible));
    GGML_ASSERT(dst->nb[0] == sizeof(T));
    GGML_ASSERT(dst->nb[1] % sizeof(T) == 0 && dst->nb[2] % sizeof(T) == 0 && dst->nb[3] % sizeof(T) == 0);

    const int64_t n_kv   = dst->ne[0];
    const int64_t n_rows = ggml_nrows(dst);

    const int64_t n_blk_x = std::min<int64_t>((n_kv + CUDA_PREFIX_MASK_BLOCK_SIZE - 1)/CUDA_PREFIX_MASK_BLOCK_SIZE, 4096);
    const int64_t n_blk_y = std::min<int64_t>(n_rows, 65535);

    const dim3 block_dims(CUDA_PREFIX_MASK_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims((unsigned) n_blk_x, (unsigned) n_blk_y, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(prefix_mask_kernel<T>, launch_params,
            (const int32_t *) n_visible->data, (T *) dst->data,
            n_kv, n_rows, dst->ne[1], dst->ne[2],
            (int64_t) (dst->nb[1]/sizeof(T)), (int64_t) (dst->nb[2]/sizeof(T)), (int64_t) (dst->nb[3]/sizeof(T)));
}

void ggml_cuda_op_prefix_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            prefix_mask_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            prefix_mask_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
