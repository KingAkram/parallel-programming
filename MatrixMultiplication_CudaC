#include <stdio.h>
#include <stdlib.h>

// Matrix multiplication kernel
__global__ void matrixMul(int *A, int *B, int *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Function to initialize matrix elements
void initMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % 10;
    }
}

// Function to print matrix elements
void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int M = 3;  // Number of rows of matrix A
    int N = 4;  // Number of columns of matrix B
    int K = 2;  // Number of columns of matrix A / Number of rows of matrix B

    // Allocate host memory
    int *h_A = (int *)malloc(M * K * sizeof(int));
    int *h_B = (int *)malloc(K * N * sizeof(int));
    int *h_C = (int *)malloc(M * N * sizeof(int));

    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Print matrices
    printf("Matrix A:\n");
    printMatrix(h_A, M, K);

    printf("Matrix B:\n");
    printMatrix(h_B, K, N);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(int));
    cudaMalloc((void **)&d_B, K * N * sizeof(int));
    cudaMalloc((void **)&d_C, M * N * sizeof(int));

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 dimGrid(ceil(N / 16.0), ceil(M / 16.0), 1);
    dim3 dimBlock(16, 16, 1);

    // Launch kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy device memory to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Matrix C:\n");
    printMatrix(h_C, M, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
