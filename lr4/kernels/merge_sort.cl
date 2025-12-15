inline int binary_search_bound(__global const int* data, int left, int right, int target, bool inclusive) {
    int mid, current_value;
    while (left < right) {
        mid = left + ((right - left) >> 1);
        current_value = data[mid];
        if (inclusive ? (current_value <= target) : (current_value < target))
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

__kernel void merge_sort(__global const int* input_data, __global int* output_data, 
                         const int total_size, int merge_step) {

    int global_idx = get_global_id(0);
    int local_idx = get_local_id(0);
    int workgroup_size = get_local_size(0);
    int workgroup_id = get_group_id(0);
    
    if (global_idx >= total_size) return; 

    int current_value = input_data[global_idx];
    int block_size = 1 << merge_step;
    int block_id = global_idx >> merge_step;
    int position_in_block = global_idx & (block_size - 1);

    int left_start, right_start, left_end, right_end;
    int count, position;

    int workgroup_start = workgroup_id * workgroup_size;
    int workgroup_end = min(workgroup_start + workgroup_size, total_size);

    if (block_id & 1) {
        left_start = block_size * (block_id - 1);
        left_end = left_start + block_size;

        position = binary_search_bound(input_data, left_start, left_end, current_value, true);
        count = position - left_start;
    } else {
        left_start = block_size * block_id;
        right_start = block_size + left_start;
        if (right_start >= total_size) {
            output_data[global_idx] = current_value;
            return;
        }
        right_end = right_start + block_size;
        if (right_end > total_size) right_end = total_size;

        position = binary_search_bound(input_data, right_start, right_end, current_value, false);
        count = position - right_start;
    }
    output_data[left_start + count + position_in_block] = current_value;
}

__kernel void array_copy(__global int* source, __global int* destination, const int size) {
    int idx = get_global_id(0);
    if (idx < size) destination[idx] = source[idx];
}
