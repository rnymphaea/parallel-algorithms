__kernel void matrix_multiply_simple(
    __global const float* A,  // Матрица A размером height x k
    __global const float* B,  // Матрица B размером k x width  
    __global float* C,        // Результирующая матрица height x width
    uint width,               // Ширина результирующей матрицы (N)
    uint height,              // Высота результирующей матрицы (M)
    uint k) {                 // Внутренняя размерность (K)
    
    // Получаем координаты текущего рабочего элемента в сетке
    uint col = get_global_id(0);  // Столбец от 0 до width-1
    uint row = get_global_id(1);  // Строка от 0 до height-1
    
    // Проверяем, что мы внутри границ результирующей матрицы
    if (col < width && row < height) {
        float sum = 0.0f;
        
        // Вычисляем скалярное произведение строки A и столбца B
        for (uint i = 0; i < k; i++) {
            // A[row, i] - элемент из строки 'row' матрицы A
            // B[i, col] - элемент из столбца 'col' матрицы B  
            sum += A[row * k + i] * B[i * width + col];
        }
        
        // Записываем результат в матрицу C
        C[row * width + col] = sum;
    }
}

// Блочное умножение матрис с использованием локальной памяти
// Определяем макросы для размеров рабочей группы
#ifndef GROUP_SIZE_X
#define GROUP_SIZE_X 16
#endif

#ifndef GROUP_SIZE_Y  
#define GROUP_SIZE_Y 16
#endif

// Требует определенного размера рабочей группы
__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
    __global const float* a,  // Матрица A: h x k
    __global const float* b,  // Матрица B: k x w
    __global float* c,        // Результирующая матрица: h x w
    uint w,                   // Ширина результата (N)
    uint h,                   // Высота результата (M) 
    uint k) {                 // Внутренняя размерность (K)
    
    __local float line_a[GROUP_SIZE_Y][GROUP_SIZE_X];  // Блок из матрицы A
    __local float line_b[GROUP_SIZE_Y][GROUP_SIZE_X];  // Блок из матрицы B

    uint global_col = get_global_id(0);  // Глобальный столбец в результирующей матрице
    uint global_row = get_global_id(1);  // Глобальная строка в результирующей матрице

    uint local_col = get_local_id(0);    // Локальный столбец внутри рабочей группы (0..GROUP_SIZE_X-1)
    uint local_row = get_local_id(1);    // Локальная строка внутри рабочей группы (0..GROUP_SIZE_Y-1)
    uint n = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X; // Количество блоков по оси K

    float s = 0.0f;
    for (uint idx = 0; idx < n; ++idx) {
        // Вычисляем индексы для загрузки данных
        uint a_col = idx * GROUP_SIZE_X + local_col; // Вычисляем индексы для загрузки данных
        uint b_row = idx * GROUP_SIZE_X + local_row; // Строка в матрице B для загрузки
        
        // Загружаем в локальную память блок из матрицы A
        if (global_row < h && a_col < k) {
            // Берем элемент A[global_row, a_col]
            line_a[local_row][local_col] = a[global_row * k + a_col];
        } else {
             // Если вышли за границы - заполняем нулями
            line_a[local_row][local_col] = 0.0f;
        }
        
        // Загружаем блок из матрицы B  
        if (b_row < k && global_col < w) {
            // Берем элемент B[b_row, global_col]
            line_b[local_row][local_col] = b[b_row * w + global_col]; 
        } else {
            line_b[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE); // Ждем пока ВСЕ потоки рабочей группы загрузят данные

        // Вычисляем частичную сумму
        for (uint i = 0; i < GROUP_SIZE_X; ++i) {
            // Умножаем строку из line_a на столбец из line_b
            s += line_a[local_row][i] * line_b[i][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE); // Ждем завершения вычислений всеми потоками
    }

    // Записываем результат
    if (global_col < w && global_row < h) {
        c[global_row * w + global_col] = s; // Записываем конечную сумму
    }
}