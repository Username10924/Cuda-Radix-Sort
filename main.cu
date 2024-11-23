#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void countingSort(int *d_input, int *d_output, int *d_countArray, int *d_positionArray, int power, int size) {
    // number of previous blocks + tidx in current block
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // increment count array based on current digit
    // atomicAdd prevents race conditions
    if (index < size) {
        int digit = (d_input[index] / power) % 10;
        atomicAdd(&d_countArray[digit], 1);
    }

    __syncthreads();

    // only need 10 threads to build the position array, for loop to emulate sequential only for this part since it's critical to be done in order
    if (index == 1) {
        for (int i = 1; i < 10; i++)
            d_positionArray[i] = d_countArray[i - 1] + d_positionArray[i - 1];
    }

    __syncthreads();

    // atomicAdd will return us the value of the position before addition
    if (index < size) {
        int digit = (d_input[index] / power) % 10;
        // increment respective postition in positionArray by 1 to setup position of next element
        int pos = atomicAdd(&d_positionArray[digit], 1);
        d_output[pos] = d_input[index];
    }
}

__global__ void findmax(int *g_idata, int *d_max) {
  extern __shared__ int sdata[];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[threadIdx.x] = g_idata[index];

  __syncthreads();
  for (int s = blockDim.x / 2; s >= 1; s = s - 2) {
    if (threadIdx.x < s && sdata[threadIdx.x] < sdata[threadIdx.x + s]) {
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  if (threadIdx.x == 0) {
    // the max integer is going to be either at index 0 or 1
    *d_max = sdata[0];
  }
}

int main() {
    
    // host memory allocations
    int input[] = {170, 45, 75, 90, 802, 24, 2, 66, 235, 45};
    int size = sizeof(input) / sizeof(input[0]);
    int output[size];
    int *max;
    max = (int*)malloc(sizeof(int));

    // device memory allocations
    int *d_input, *d_output, *d_countArray, *d_positionArray, *d_max;
    cudaMalloc((void **) &d_input, size * sizeof(int));
    cudaMalloc((void **) &d_output, size * sizeof(int));
    cudaMalloc((void **) &d_countArray, 10 * sizeof(int));
    cudaMalloc((void **) &d_positionArray, 10 * sizeof(int));
    cudaMalloc((void **) &d_max, sizeof(int));

    // configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock) / threadsPerBlock;

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    findmax<<<blocksPerGrid, size, threadsPerBlock>>>(d_input, d_max);
    cudaMemcpy(max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
        cout << *max << "\n";

    int power = 1;

    while (*max / power > 0) {
        // reset arrays for next iteration
        cudaMemset(d_countArray, 0, 10 * sizeof(int));
        cudaMemset(d_positionArray, 0, 10 * sizeof(int));

        countingSort<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_countArray, d_positionArray, power, size);

        // copy previous values to next iteration to calculate new values, hence, from device memory to device memory
        cudaMemcpy(d_input, d_output, size * sizeof(int), cudaMemcpyDeviceToDevice);

        // next least significant digit
        power *= 10;
    }

    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_countArray);
    cudaFree(d_positionArray);

    cout << "result: ";
    for (int i = 0; i < size; i++) {
        cout << output[i] << " ";
    }

    return 0;
}