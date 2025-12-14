inline int upper_bound_cmp(__global const int* arr, int l, int r, int val, bool in) {
    int mid, v;
    while (l < r) {
        mid = l + ((r - l) >> 1);
        v = arr[mid];
        if (in ? (v <= val) : (v < val))
            l = mid + 1;
        else
            r = mid;
    }
    return l;
}

//last_layer - исходный массив (результат предыдущей итерации)
//new_layer - новый массив для записи результата
//n - размер массива
//pow - текущая "степень" (определяет размер блоков для слияния)

__kernel void merge_sort(__global const int* last_layer, __global int* new_layer, 
                         const int n, int pow) {

    int global_id = get_global_id(0);  // Глобальный ID потока (0..n-1)
    int local_id = get_local_id(0);    // Локальный ID внутри work group (0..work_group_size-1)
    int group_size = get_local_size(0); // Размер work group (из конфига)
    int group_id = get_group_id(0);    // ID work group
    
    if (global_id >= n) return; 

    int val = last_layer[global_id];
    int block_size = 1 << pow;                                                                                     
    int block_id = global_id >> pow;  // Номер блока = global_id / block_size                                                    
    int pos_in_block = global_id & (block_size - 1); // Позиция в блоке = global_id % block_size

    int l_start, r_start, l_end, r_end;
    int cnt, pos;

    // Оптимизация: используем групповой доступ к памяти
    // Потоки в одной группе обрабатывают смежные данные для лучшей cache locality
    int group_start = group_id * group_size;
    int group_end = min(group_start + group_size, n);

    //Случай A: Нечетный блок (объединяется с левым блоком)
    //Элемент из нечетного блока ищет свою позицию в левом (четном) блоке.
    if (block_id & 1) {
        l_start = block_size * (block_id - 1);
        l_end = l_start + block_size;

        pos = upper_bound_cmp(last_layer, l_start, l_end, val, true);
        cnt = pos - l_start;
    } else 
    //Случай B: Четный блок (объединяется с правым блоком)
    //Элемент из четного блока ищет, сколько элементов из правого блока должны быть перед ним.
    {
        l_start = block_size * block_id;
        r_start = block_size + l_start;
        if (r_start >= n) {
            new_layer[global_id] = val;
            return;
        }
        r_end = r_start + block_size;
        if (r_end > n) r_end = n;

        pos = upper_bound_cmp(last_layer, r_start, r_end, val, false);
        cnt = pos - r_start;
    }
    new_layer[l_start + cnt + pos_in_block] = val;
}

__kernel void copy_array(__global int* src, __global int* dst, const int size) {
    int idx = get_global_id(0);
    if (idx < size) dst[idx] = src[idx];
}