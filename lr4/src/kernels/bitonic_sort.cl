__kernel void bitonic_sort(
    __global float* data,
    const uint size,
    const uint stage,
    const uint passOfStage)
{
    uint i = get_global_id(0) * 2;
    if (i >= size) return;
    
    uint direction = ((i / (1 << stage)) % 2) == 0 ? 1 : 0;
    uint j = i ^ (1 << (stage - passOfStage));
    
    if (j > i) {
        float a = data[i];
        float b = data[j];
        
        if ((direction == 1 && a > b) || (direction == 0 && a < b)) {
            data[i] = b;
            data[j] = a;
        }
    }
}
