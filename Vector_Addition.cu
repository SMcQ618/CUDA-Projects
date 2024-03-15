#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>
#define N 1024


//Error checking function
void checkCudaError(cudaError_t result, char const *const func, const char *const file, int const line) {
        if (result != cudaSuccess) {
                std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " (" << cudaGetErrorName(result) << ") " << "\"" << std:: endl;
                //make the program exit with the error
                exit(EXIT_FAILURE);
        }
}

//Macro for the error checking
#define CUDA_CHECK(val) checkCudaError((val), #val, __FILE__, __LINE__)


//CUDA Kernel
__global__ void vector_add(float *a, float *b, float *result, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
                result[idx] = a[idx] + b[idx];
        }
}

//main function
int main() {

        size_t bytes = N * sizeof(float);

        //need to allocate the host memory
        float *h_a, *h_b, *h_result;
        h_a = (float*)malloc(N * sizeof(float));
        h_b = (float*)malloc(N * sizeof(float));
        h_result = (float*)malloc(N * sizeof(float));

        for(int i = 0; i < N; ++i) {
                h_a[i] = 1.0f;
                h_b[i] = 2.0f;
        }

        //make sure device memory is not bad
        float *d_a, *d_b, *d_result;
        CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_result, bytes));

        //get data from host to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

        //get the grid
        dim3 blockSize(256);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

        //launch the kernel
        vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_result, N);
        CUDA_CHECK(cudaGetLastError()); //chceck for the kernel errors

        //copy the result from device to host
        CUDA_CHECK(cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost));

        //Dislpay result or further processing
        for(int i = 0; i < N; ++i) {
                std::cout << h_result[i] << " ";      
        }
        std::cout << std::endl;
        //free the device and host memory
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_result));
        //release the host memeory
        free(h_a);
        free(h_b);
        free(h_result);
        return 0;
}
