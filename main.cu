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

    // only the first thread in the block builds positionArray
    if (index == 0) {
        for (int i = 1; i < 10; i++)
            d_positionArray[i] = d_countArray[i - 1] + d_positionArray[i - 1];
    }

    __syncthreads();
    // atomicAdd will return us the value of the position before addition
    // threads in different warps editing values at the same time cause a problem of unstable sorting, we have to sort warp by warp
    for(int i = 32; i <= size; i += 32) {
        if (index < i && index >= i - 32) {
            int digit = (d_input[index] / power) % 10;
            // increment respective postition in positionArray by 1 to setup position of next element
            int pos = atomicAdd(&d_positionArray[digit], 1);
            d_output[pos] = d_input[index];
        }
        __syncthreads();
    }
    
}

__global__ void findmax(int *g_idata, int *d_max) {
  extern __shared__ int sdata[];

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  sdata[threadIdx.x] = g_idata[index];

  __syncthreads();
  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (threadIdx.x < s && sdata[threadIdx.x] < sdata[threadIdx.x + s]) {
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  if (threadIdx.x == 0) {
    *d_max = sdata[0];
  }
}

int main() {
    srand(time(0));
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // host memory allocations
    int size = 1024;
    int *input = (int*)malloc(size * sizeof(int));
    int *output = (int*)malloc(size * sizeof(int));
    for(int i = 0; i < 1024; i++) {
        input[i] = rand() % 1000;
    }
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
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    findmax<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_max);
    cudaDeviceSynchronize();
    cudaMemcpy(max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    int power = 1;
    // start timer
    cudaEventRecord(start);
    while (*max / power > 0) {
        // reset arrays for next iteration
        cudaMemset(d_countArray, 0, 10 * sizeof(int));
        cudaMemset(d_positionArray, 0, 10 * sizeof(int));

        countingSort<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_countArray, d_positionArray, power, size);
        cudaDeviceSynchronize();
        cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_input, output, size * sizeof(int), cudaMemcpyHostToDevice);
        
        // next least significant digit
        power *= 10;
    }
    // end timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000;
    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_countArray);
    cudaFree(d_positionArray);

    cout << "result: ";
    for (int i = 0; i < size; i++) {
        cout << output[i] << " ";
    }
    cout << "\n";
    cout <<"TIME TAKEN TO SORT " << size << " ELEMENTS: " << seconds << " seconds";

    return 0;
}