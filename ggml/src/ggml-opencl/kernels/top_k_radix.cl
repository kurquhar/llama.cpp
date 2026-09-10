#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)

static inline uint float_to_key(float value) {
    const uint bits = as_uint(value);
    return (bits & 0x80000000u) ? bits ^ 0xffffffffu : bits | 0x80000000u;
}

kernel void kernel_top_k_radix_f32_i32(
    global const char * src0,
    ulong               offset0,
    global char *       dst,
    ulong               offsetd,
    int                 ncols,
    int                 top_k
) {
    const int tid = get_local_id(0);
    const int row = get_group_id(1);

    global const float * src_row = (global const float *) (src0 + offset0) + row * ncols;
    global int * dst_row = (global int *) (dst + offsetd) + row * top_k;

    local uint histogram[RADIX_SIZE];
    local uint bucket;
    local uint above;
    local uint out_count;

    uint prefix = 0;
    uint remaining = top_k;

    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        if (tid < RADIX_SIZE) {
            histogram[tid] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const uint high_mask = shift + RADIX_BITS >= 32 ? 0u : 0xffffffffu << (shift + RADIX_BITS);
        const uint high_prefix = prefix & high_mask;
        for (int col = tid; col < ncols; col += get_local_size(0)) {
            const uint key = float_to_key(src_row[col]);
            if ((key & high_mask) == high_prefix) {
                atomic_inc(&histogram[(key >> shift) & (RADIX_SIZE - 1)]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid == 0) {
            uint accumulated = 0;
            uint selected_bucket = 0;
            for (int current = RADIX_SIZE - 1; current >= 0; --current) {
                const uint count = histogram[current];
                if (accumulated + count >= remaining) {
                    selected_bucket = current;
                    break;
                }
                accumulated += count;
            }
            bucket = selected_bucket;
            above = accumulated;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        prefix |= bucket << shift;
        remaining -= above;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        out_count = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int col = tid; col < ncols; col += get_local_size(0)) {
        if (float_to_key(src_row[col]) > prefix) {
            dst_row[atomic_inc(&out_count)] = col;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int col = tid; col < ncols; col += get_local_size(0)) {
        if (float_to_key(src_row[col]) == prefix) {
            const uint output = atomic_inc(&out_count);
            if (output < top_k) {
                dst_row[output] = col;
            }
        }
    }
}
