#include <cstdio>
#include <cinttypes>
#include <chrono>
#include <bitset>

#define BLOCK_SIZE 64

// Задача 1. Определить бит для каждой нити с учетом того, что каждой нити прихоидтся N (около 3-5, больше 5 может отсутствовать какой-либо выигрыш) чисел
// Задача 2. Расчитать АКФ для каждого числа
// Задача 3. Запись результата в глобальную память одна на блок, наилучший результат
//
// Можно использовать атомики чтобы писать в глобальную память только одно значение
// Начальный пик всегда ярко выражен, поэтому его учитывать не надо
// Париться о длинной арифметике не надо - больше 32-33 бит посчитать не получится в адекватное время

__global__ void akf(uint64_t start_offset, uint64_t end, uint64_t *min_amplitude, uint64_t *signal, uint64_t n) {
    __shared__ int8_t akf[BLOCK_SIZE];
    __shared__ uint8_t bits[BLOCK_SIZE];
    __shared__ uint64_t amplitude;
    size_t idx = threadIdx.x; // offset
    uint64_t val = blockIdx.x + start_offset; // signal
    while (val <= end) {
        amplitude = 0;
        bits[n - idx - 1] = (val >> idx) & 1;
        __syncthreads();

        int8_t out = 0;
        for (size_t i = 0; i < n - idx; i++) {
            out -= (bits[i + idx] ^ bits[i]) * 2 - 1;
        }

        akf[idx] = (out >= 0) ? out : 0 - out;
        if (idx != 0) {
            atomicMax(reinterpret_cast<unsigned long long int*>(&amplitude), (unsigned long long)akf[idx]);
        }
        __syncthreads();

        if (idx == 0) {
            uint64_t old = atomicMin(reinterpret_cast<unsigned long long int*>(min_amplitude), (unsigned long long)amplitude);
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

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage: analysis [size]\n");
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]);
    printf("Using %d bits\n", n);
    if(n >= 64) {
        printf("Cannot process more than 63 bits\n");
        return EXIT_FAILURE;
    }
    if(n < 5) {
        printf("Cannot process less than 5 bits\n");
        return EXIT_FAILURE;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t *dev_amplitude;
    uint64_t *dev_signal;
    cudaMalloc((void**)&dev_amplitude, sizeof(uint64_t));
    cudaMalloc((void**)&dev_signal, sizeof(uint64_t));
    cudaMemset(dev_amplitude, 0xFF, sizeof(uint64_t));
    uint64_t amplitude;
    uint64_t signal;

    uint64_t blocks = (1ULL << n) - 1ULL;
    uint64_t start = 0b1ULL << (n - 1);
    akf<<<3072, n>>>(start, blocks, dev_amplitude, dev_signal, n);
    cudaError_t cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Could not execute CUDA kernel: %s\n", cudaGetErrorString(cudaError));
        return EXIT_FAILURE;
    }
    cudaMemcpy(&amplitude, dev_amplitude, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&signal, dev_signal, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    std::bitset<BLOCK_SIZE> s(signal);
    printf("Best signal is %s (%llu) with amplitude of %llu\n", s.to_string().c_str(), signal, amplitude);

    auto *akf = new int8_t[n]();
    cpu_akf(signal, n, akf);
    printf("AKF is ");
    for(int8_t i = 0; i < n; i++) {
        printf("%d ", akf[i]);
    }
    printf("\n");
    auto end_time = std::chrono::high_resolution_clock::now();
    printf("Calculation took %f seconds", (double)(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1e+6);
    return 0;
}
