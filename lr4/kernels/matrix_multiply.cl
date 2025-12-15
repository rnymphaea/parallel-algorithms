__kernel void matrix_multiply_simple(
    __global const float* matrix_a,
    __global const float* matrix_b,
    __global float* result_matrix,
    uint matrix_width,
    uint matrix_height,
    uint inner_dimension) {
    
    uint column = get_global_id(0);
    uint row = get_global_id(1);
    
    if (column < matrix_width && row < matrix_height) {
        float sum = 0.0f;
        
        for (uint i = 0; i < inner_dimension; i++) {
            sum += matrix_a[row * inner_dimension + i] * matrix_b[i * matrix_width + column];
        }
        
        result_matrix[row * matrix_width + column] = sum;
    }
}

#ifndef GROUP_SIZE_X
#define GROUP_SIZE_X 16
#endif

#ifndef GROUP_SIZE_Y  
#define GROUP_SIZE_Y 16
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_multiply_blocked(
    __global const float* matrix_a,
    __global const float* matrix_b,
    __global float* result_matrix,
    uint matrix_width,
    uint matrix_height,
    uint inner_dimension) {
    
    __local float local_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float local_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    uint global_column = get_global_id(0);
    uint global_row = get_global_id(1);

    uint local_column = get_local_id(0);
    uint local_row = get_local_id(1);
    uint block_count = (inner_dimension + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    float accumulator = 0.0f;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        uint a_column = block_index * GROUP_SIZE_X + local_column;
        uint b_row = block_index * GROUP_SIZE_X + local_row;
        
        if (global_row < matrix_height && a_column < inner_dimension) {
            local_a[local_row][local_column] = matrix_a[global_row * inner_dimension + a_column];
        } else {
            local_a[local_row][local_column] = 0.0f;
        }
        
        if (b_row < inner_dimension && global_column < matrix_width) {
            local_b[local_row][local_column] = matrix_b[b_row * matrix_width + global_column];
        } else {
            local_b[local_row][local_column] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < GROUP_SIZE_X; ++i) {
            accumulator += local_a[local_row][i] * local_b[i][local_column];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_column < matrix_width && global_row < matrix_height) {
        result_matrix[global_row * matrix_width + global_column] = accumulator;
    }
}
