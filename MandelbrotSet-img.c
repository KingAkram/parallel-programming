#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_ITER 1000
#define TAG_REGION 1
#define TAG_RESULT 2

typedef struct {
    double x;
    double y;
} complex;

// Calculate the Mandelbrot set for a given pixel
int mandelbrot(complex c) {
    int i;
    complex z;
    z.x = 0;
    z.y = 0;
    for (i = 0; i < MAX_ITER; i++) {
        complex z_next;
        z_next.x = z.x * z.x - z.y * z.y + c.x;
        z_next.y = 2 * z.x * z.y + c.y;
        if (z_next.x * z_next.x + z_next.y * z_next.y > 4) {
            return i;
        }
        z = z_next;
    }
    return MAX_ITER;
}

// Write the image to a PPM file
void write_image(int *image, char *filename) {
    FILE *fp;
    int i, j;
    fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    for (j = 0; j < HEIGHT; j++) {
        for (i = 0; i < WIDTH; i++) {
            unsigned char pixel[3];
            pixel[0] = (unsigned char)(image[j * WIDTH + i] * 255 / MAX_ITER);
            pixel[1] = (unsigned char)(image[j * WIDTH + i] * 255 / MAX_ITER);
            pixel[2] = (unsigned char)(image[j * WIDTH + i] * 255 / MAX_ITER);
            fwrite(pixel, 1, 3, fp);
        }
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int *image = (int *)malloc(sizeof(int) * WIDTH * HEIGHT);
    int rows_per_process = HEIGHT / size;
    
    // Static method
    if (rank == 0) {
        printf("Static method\n");
        double start_time = MPI_Wtime();
        int i, j, k;
        for (i = 0; i < size; i++) {
            int start_row = i * rows_per_process;
            int end_row = (i + 1) * rows_per_process;
            if (i == size - 1) {
                end_row = HEIGHT;
            }
            // Send the region to the Worker
            for (j = start_row; j < end_row; j++) {
                MPI_Send(&j, 1, MPI_INT, i, TAG_REGION, MPI_COMM_WORLD);
                for (k = 0; k < WIDTH; k++) {
                    complex c;
                    c.x = (k - WIDTH / 2.0) * 4.0 / WIDTH;
                    c.y = (j - HEIGHT / 2.0) * 4.0 / WIDTH;
                    MPI_Send(&c, sizeof(complex), MPI_BYTE, i, TAG_REGION, MPI_COMM_WORLD);
                }
            }
        }
        
        // Receive the results from the Workers and combine them
        int row;
        for (row = 0; row < HEIGHT; row++) {
            int source = row / rows_per_process;
            int offset = row % rows_per_process;
            if (source == 0) {
                int i;
                for (i = 0; i < WIDTH; i++) {
                    image[row * WIDTH + i] = mandelbrot((complex){(i - WIDTH / 2.0) * 4.0 / WIDTH, (row - HEIGHT / 2.0) * 4.0 / WIDTH});
                }
            } else {
                MPI_Recv(&image[row * WIDTH], WIDTH, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        double end_time = MPI_Wtime();
        printf("Time taken: %f seconds\n", end_time - start_time);
        write_image(image, "mandelbrot_static.ppm");
    }
    
    // Worker
    else {
        int row;
        for (;;) {
            MPI_Recv(&row, 1, MPI_INT, 0, TAG_REGION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (row == -1) break;
            int i;
            for (i = 0; i < WIDTH; i++) {
                complex c;
                MPI_Recv(&c, sizeof(complex), MPI_BYTE, 0, TAG_REGION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                image[(row - rank * rows_per_process) * WIDTH + i] = mandelbrot(c);
            }
            MPI_Send(&image[(row - rank * rows_per_process) * WIDTH], WIDTH, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }
    
    // Dynamic method
    if (rank == 0) {
        printf("Dynamic method\n");
        double start_time = MPI_Wtime();
        int i, j;
        for (i = 0; i < HEIGHT; i++) {
            int row = i;
            int source, offset;
            if (row < size * rows_per_process) {
                source = row / rows_per_process;
                offset = row % rows_per_process;
                MPI_Send(&row, 1, MPI_INT, source, TAG_REGION, MPI_COMM_WORLD);
                for (j = 0; j < WIDTH; j++) {
                    complex c;
                    c.x = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                    c.y = (row - HEIGHT / 2.0) * 4.0 / WIDTH;
                    MPI_Send(&c, sizeof(complex), MPI_BYTE, source, TAG_REGION, MPI_COMM_WORLD);
                }
            }
            MPI_Recv(&image[row * WIDTH], WIDTH, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (i = 1; i < size; i++) {
            int row = -1;
            MPI_Send(&row, 1, MPI_INT, i, TAG_REGION, MPI_COMM_WORLD);
        }
        for (i = 0; i < size * rows_per_process; i++) {
            int row;
            MPI_Recv(&row, 1, MPI_INT, MPI_ANY_SOURCE, TAG_REGION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int source = row / rows_per_process;
            int offset = row % rows_per_process;
            MPI_Recv(&image[row * WIDTH], WIDTH, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (i + size * rows_per_process < HEIGHT) {
                row = i + size * rows_per_process;
                source = row / rows_per_process;
                offset = row % rows_per_process;
                MPI_Send(&row, 1, MPI_INT, source, TAG_REGION, MPI_COMM_WORLD);
                for (j = 0; j < WIDTH; j++) {
                    complex c;
                    c.x = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                    c.y = (row - HEIGHT / 2.0) * 4.0 / WIDTH;
                    MPI_Send(&c, sizeof(complex), MPI_BYTE, source, TAG_REGION, MPI_COMM_WORLD);
                }
            }
        }
        double end_time = MPI_Wtime();
        printf("Time taken: %f seconds\n", end_time - start_time);
        write_image(image, "mandelbrot_dynamic.ppm");
    }
    // Worker
    else {
        for (;;) {
            int row;
            MPI_Recv(&row, 1, MPI_INT, 0, TAG_REGION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (row == -1) break;
            int i;
            for (i = 0; i < WIDTH; i++) {
                complex c;
                MPI_Recv(&c, sizeof(complex), MPI_BYTE, 0, TAG_REGION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                image[(row - rank * rows_per_process) * WIDTH + i] = mandelbrot(c);
            }
            MPI_Send(&image[(row - rank * rows_per_process) * WIDTH], WIDTH, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    free(image);
    
    return 0;
}
