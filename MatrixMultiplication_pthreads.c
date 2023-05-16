#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_THREADS 100

int M, N, K;
int matrix_A[MAX_THREADS][MAX_THREADS];
int matrix_B[MAX_THREADS][MAX_THREADS];
int matrix_C[MAX_THREADS][MAX_THREADS];

void *multiply(void *arg) {
    int thread_id = *((int *)arg);

    int start_row = (thread_id * M) / K;
    int end_row = ((thread_id + 1) * M) / K;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            matrix_C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }

    pthread_exit(NULL);
}

int main() {
    int num_threads;
    printf("Enter the number of threads (1-%d): ", MAX_THREADS);
    scanf("%d", &num_threads);

    if (num_threads <= 0 || num_threads > MAX_THREADS) {
        printf("Invalid number of threads. Exiting...\n");
        return 1;
    }

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
            scanf("%d", &matrix_A[i][j]);
        }
    }

    printf("Enter the elements of matrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            scanf("%d", &matrix_B[i][j]);
        }
    }

    pthread_t threads[MAX_THREADS];
    int thread_ids[MAX_THREADS];

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, multiply, (void *)&thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Resultant matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%d ", matrix_C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
