#include <cstdio>
#include <cinttypes>

#define BLOCK_SIZE 25

// Задача 1. Определить бит для каждой нити с учетом того, что каждой нити прихоидтся N (около 3-5, больше 5 может отсутствовать какой-либо выигрыш) чисел
// Задача 2. Расчитать АКФ для каждого числа
// Задача 3. Запись результата в глобальную память одна на блок, наилучший результат
//
// Можно использовать атомики чтобы писать в глобальную память только одно значение
// Начальный пик всегда ярко выражен, поэтому его учитывать не надо
// Париться о длинной арифметике не надо - больше 32-33 бит посчитать не получится в адекватное время

__global__ void akf(uint64_t start_offset, int8_t *output) {
    __shared__ int8_t akf[BLOCK_SIZE];
    __shared__ uint8_t bits[BLOCK_SIZE];
    size_t idx = threadIdx.x; // offset
    uint64_t val = blockIdx.x + start_offset; // signal

    bits[BLOCK_SIZE - idx - 1] = (val >> idx) & 1;
    __syncthreads();

    int8_t out = 0;
    for(size_t i = 0; i < BLOCK_SIZE - idx; i++) {
        out -= (bits[i + idx] ^ bits[i]) * 2 - 1;
    }

    akf[idx] = out;
    __syncthreads();

    if(idx == 0) {
        for(size_t i = 0; i < BLOCK_SIZE; i++) {
            output[blockIdx.x * BLOCK_SIZE + i] = akf[i];
        }
    }
}

int main() {
    int8_t *dev_output;
    cudaMalloc((void**)&dev_output, BLOCK_SIZE);
    auto output = new int8_t[BLOCK_SIZE];
    akf<<<1, BLOCK_SIZE>>>(0b1110011100000010101001001, dev_output);
    cudaMemcpy(output, dev_output, BLOCK_SIZE, cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < BLOCK_SIZE; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
    return 0;
}
