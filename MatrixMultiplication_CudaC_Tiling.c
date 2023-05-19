#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

// Matrix multiplication kernel with tiling
__global__ void matrixMul(int *A, int *B, int *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int sum = 0;
    int numTiles = ceil((float)K / TILE_SIZE);

    for (int t = 0; t < numTiles; t++) {
        int tileRow = blockIdx.y * blockDim.y + threadIdx.y;
        int tileCol = t * TILE_SIZE + threadIdx.x;

        if (tileRow < M && tileCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[tileRow * K + tileCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        tileRow = t * TILE_SIZE + threadIdx.y;
        tileCol = blockIdx.x * blockDim.x + threadIdx.x;

        if (tileRow < K && tileCol < N) {
            tileB[threadIdx.y][threadIdx.x] = B[tileRow * N + tileCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
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
    dim3 dimGrid(ceil(N / (float)TILE_SIZE), ceil(M / (float)TILE_SIZE), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

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
