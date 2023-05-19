#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SIZE 100

void matrix_multiply(int A[MAX_SIZE][MAX_SIZE], int B[MAX_SIZE][MAX_SIZE], int C[MAX_SIZE][MAX_SIZE], int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int M, N, K;
    int A[MAX_SIZE][MAX_SIZE], B[MAX_SIZE][MAX_SIZE], C[MAX_SIZE][MAX_SIZE];

    printf("Enter the dimensions of matrix A (M N): ");
    scanf("%d %d", &M, &N);

    printf("Enter the dimensions of matrix B (N K): ");
    scanf("%d %d", &N, &K);

    if (M <= 0 || N <= 0 || K <= 0) {
        printf("Invalid matrix dimensions. Exiting...\n");
        return 1;
    }

    printf("Enter the elements of matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &A[i][j]);
        }
    }

    printf("Enter the elements of matrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            scanf("%d", &B[i][j]);
        }
    }

    matrix_multiply(A, B, C, M, N, K);

    printf("Resultant matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
