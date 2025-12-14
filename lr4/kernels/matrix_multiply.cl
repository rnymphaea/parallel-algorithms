__kernel void matrix_multiply_simple(
    __global const float* A,
    __global const float* B,
    __global float* C,
    uint width,
    uint height,
    uint k) {
    
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    
    if (col < width && row < height) {
        float sum = 0.0f;
        
        for (uint i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * width + col];
        }
        
        C[row * width + col] = sum;
    }
}

#ifndef GROUP_SIZE_X
#define GROUP_SIZE_X 16
#endif

#ifndef GROUP_SIZE_Y  
#define GROUP_SIZE_Y 16
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
    __global const float* a,
    __global const float* b,
    __global float* c,
    uint w,
    uint h,
    uint k) {
    
    __local float line_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float line_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    uint global_col = get_global_id(0);
    uint global_row = get_global_id(1);

    uint local_col = get_local_id(0);
    uint local_row = get_local_id(1);
    uint n = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    float s = 0.0f;
    for (uint idx = 0; idx < n; ++idx) {
        uint a_col = idx * GROUP_SIZE_X + local_col;
        uint b_row = idx * GROUP_SIZE_X + local_row;
        
        if (global_row < h && a_col < k) {
            line_a[local_row][local_col] = a[global_row * k + a_col];
        } else {
            line_a[local_row][local_col] = 0.0f;
        }
        
        if (b_row < k && global_col < w) {
            line_b[local_row][local_col] = b[b_row * w + global_col];
        } else {
            line_b[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < GROUP_SIZE_X; ++i) {
            s += line_a[local_row][i] * line_b[i][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_col < w && global_row < h) {
        c[global_row * w + global_col] = s;
    }
}
