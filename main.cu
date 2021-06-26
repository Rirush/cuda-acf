#include <cstdio>
#include <cinttypes>

#define BLOCK_SIZE 13

__global__ void akf(uint64_t start_offset, uint8_t *output) {
    __shared__ uint8_t akf[BLOCK_SIZE];
    size_t idx = threadIdx.x; // offset
    uint64_t val = blockIdx.x + start_offset; // signal

    uint8_t bits[BLOCK_SIZE];
    for(uint8_t i = 0; i < BLOCK_SIZE; i++) {
        bits[BLOCK_SIZE - i - 1] = val & (1);
        val >>= 1;
    }

    uint8_t out = 0;
    for(size_t i = 0; i < BLOCK_SIZE - idx; i++) {
        out += bits[i + idx] & bits[i];
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
    uint8_t *dev_output;
    cudaMalloc((void**)&dev_output, BLOCK_SIZE);
    auto output = new uint8_t[BLOCK_SIZE];
    akf<<<1, BLOCK_SIZE>>>(0b1111100110101, dev_output);
    cudaMemcpy(output, dev_output, BLOCK_SIZE, cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < BLOCK_SIZE; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
    return 0;
}
