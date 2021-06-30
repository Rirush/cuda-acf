#include <cstdio>
#include <cinttypes>
#include <chrono>

#define BLOCK_SIZE 35

// Задача 1. Определить бит для каждой нити с учетом того, что каждой нити прихоидтся N (около 3-5, больше 5 может отсутствовать какой-либо выигрыш) чисел
// Задача 2. Расчитать АКФ для каждого числа
// Задача 3. Запись результата в глобальную память одна на блок, наилучший результат
//
// Можно использовать атомики чтобы писать в глобальную память только одно значение
// Начальный пик всегда ярко выражен, поэтому его учитывать не надо
// Париться о длинной арифметике не надо - больше 32-33 бит посчитать не получится в адекватное время

__global__ void akf(uint64_t start_offset, uint64_t end, uint64_t *min_amplitude, uint64_t *signal, uint64_t *counter) {
    __shared__ int8_t akf[BLOCK_SIZE];
    __shared__ uint8_t bits[BLOCK_SIZE];
    __shared__ uint64_t amplitude;
    size_t idx = threadIdx.x; // offset
    uint64_t val = blockIdx.x + start_offset; // signal
    while (val <= end) {
        amplitude = 0;
        bits[BLOCK_SIZE - idx - 1] = (val >> idx) & 1;
        __syncthreads();

        int8_t out = 0;
        for (size_t i = 0; i < BLOCK_SIZE - idx; i++) {
            out -= (bits[i + idx] ^ bits[i]) * 2 - 1;
        }

        akf[idx] = (out >= 0) ? out : 0 - out;
        if (idx != 0) {
            atomicMax(&amplitude, akf[idx]);
        }
        __syncthreads();

        if (idx == 0) {
            uint64_t old = atomicMin(min_amplitude, amplitude);
            if (old >= amplitude) {
                *signal = val;
            }
        }

        val += gridDim.x;
        __syncthreads();
    };
}

void cpu_akf(uint64_t value, uint8_t b, int8_t *akf) {
    auto bits = new int8_t[b]();
    for(size_t i = 0; i < b; i++) {
        int8_t bit = (value & 1) ? 1 : -1;
        bits[b - i - 1] = bit;
        value >>= 1;
    }

    for(size_t offset = 0; offset < b; offset++) {
        for(size_t pos = 0; pos + offset < b; pos++) {
            akf[offset] += bits[pos] * bits[pos + offset];
        }
    }
    delete[] bits;
}

int main() {
    std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::high_resolution_clock::now();
    uint64_t *dev_amplitude;
    uint64_t *dev_signal;
    uint64_t *dev_counter;
    cudaMalloc((void**)&dev_amplitude, sizeof(uint64_t));
    cudaMalloc((void**)&dev_signal, sizeof(uint64_t));
    cudaMalloc((void**)&dev_counter, sizeof(uint64_t));
    cudaMemset(dev_amplitude, 0xFF, sizeof(uint64_t));
    cudaMemset(dev_counter, 0x00, sizeof(uint64_t));
    uint64_t amplitude;
    uint64_t signal;
    uint64_t counter;

    uint64_t blocks = (1ULL << BLOCK_SIZE) - 1ULL;
    uint64_t start = 0b1ULL << (BLOCK_SIZE - 1);
    akf<<<2048, BLOCK_SIZE>>>(start, blocks, dev_amplitude, dev_signal, dev_counter);
    cudaError_t cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Could not execute CUDA kernel: %s\n", cudaGetErrorString(cudaError));
        return EXIT_FAILURE;
    }
    cudaMemcpy(&amplitude, dev_amplitude, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&signal, dev_signal, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&counter, dev_counter, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    char buf[66];
    char mask[66];
    _ui64toa(signal, buf, 2);
    _ui64toa(counter, mask, 2);
    printf("Best signal is %s (%llu) with amplitude of %llu\n", buf, signal, amplitude);
    printf("Processed mask: %s\n", mask);

    auto *akf = new int8_t[BLOCK_SIZE]();
    cpu_akf(signal, BLOCK_SIZE, akf);
    printf("AKF is ");
    for(int8_t i = 0; i < BLOCK_SIZE; i++) {
        printf("%d ", akf[i]);
    }
    printf("\n");
    std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();
    printf("Calculation took %f seconds", (double)(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1e+6);
    return 0;
}
