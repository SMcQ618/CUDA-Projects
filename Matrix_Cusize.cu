#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>
//create the kernel
__global__ void matrixMat(float *A, float *B, float *C, int N, int blockSize)
{
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < N && col < N)
        {
                float sum = 0.0f;
                //start the matrix multiplication things
                for (int k = 0; k < N; k += blockSize)
                {
                        for(int i = 0; i < blockSize && (k + i) < N; ++i)
                        {
                                sum += A[row * N + (k + i)] * B[(k + i) * N + col];
                        }
                }
                //store the result in the output of C
                C[row * N + col] = sum;
        }
        //probably should include some error checking
}
//error checking
void checkCudaError(cudaError_t result, char const *const func, const char *const file, int const line) {
        if(result != cudaSuccess) {
                fprintf(stderr, "CUDA error at %s:%d code = %d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        //make it exit with the error
        exit(EXIT_FAILURE);
        }
}
//do the main stuff here
int main()
{
        const int N = 1024;
        //set the block size (i think this will be the thing that is being changed
        //make the size in powers of 2
        int blockSize = 512;
        //allocate the ememry on the host
        float *h_A = (float*)malloc(N * N * sizeof(float));
        float *h_B = (float*)malloc(N * N * sizeof(float));
        float *h_C = (float*)malloc(N * N * sizeof(float));
         
        //initiate teh matricies of A and B with 1 and 2
        for (int i = 0; i < N * N; ++i)
        {
                h_A[i] = 1.0f;
                h_B[i] = 2.0f; 
        }
        //make space for the memory on teh device       
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float)); 
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        
        //copy the values of the matricies from the host to device
        cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);   
        //make the grid and block dimensions
        //the block will be the threads per block
        dim3 blockDim(blockSize, blockSize);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
        //initiate the kernel
        matrixMat<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, blockSize);
        //give some time for everything to be copied before actualy coping everything   
        cudaDeviceSynchronize();
        //copy the result of C from the device to the host
        cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyHostToDevice);
        //print the result
        printf("Result of C:\n");
        for(int i = 0; i < N; ++i)
        {
                for(int j = 0; j < N; ++j)
                {
                        printf("%.1f\t", h_C[i * N + j]);
                }
                printf("\n");
        }
        //free the device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        //free the host memory
        free(h_A);
        free(h_B);
        free(h_C);
        return 0;
}
