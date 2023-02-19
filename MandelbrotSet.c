#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define WIDTH 78
#define HEIGHT 44
#define MAX_ITER 10000

void compute_mandelbrot(int start, int end, int* data);

int main(int argc, char** argv)
{
    int rank, size, i, j, iter, remainder, chunk_size, chunk_start, chunk_end;
    int* data;
    int* counts;
    int* displs;
    MPI_Status status;
    
    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // calculate the portion of the image that each process is responsible for
    chunk_size = (HEIGHT * WIDTH) / size;
    remainder = (HEIGHT * WIDTH) % size;
    
    // allocate memory for the data array
    data = (int*)malloc(chunk_size * sizeof(int));
    
    // compute the Mandelbrot set for the portion of the image assigned to this process
    chunk_start = rank * chunk_size;
    if (rank < remainder)
    {
        chunk_size++;
        chunk_start += rank;
    }
    else
    {
        chunk_start += remainder;
    }
    chunk_end = chunk_start + chunk_size;
    
    // calculate the Mandelbrot set using the selected load balancing method
    if (argc > 1 && strcmp(argv[1], "dynamic") == 0)
    {
        compute_mandelbrot(chunk_start, chunk_end, data);
    }
    else
    {
        // staticstatic load balancing: divide the image into strips and assign each process a strip
        for (i = chunk_start; i < chunk_end; i++)
        {
            for (j = 0; j < WIDTH; j++)
            {
                double x = ((double)j - (double)WIDTH / 2.0) * 4.0 / (double)WIDTH;
                double y = ((double)i - (double)HEIGHT / 2.0) * 4.0 / (double)WIDTH;
                double cx = x;
                double cy = y;
                iter = 0;
                while (cx * cx + cy * cy < 4 && iter < MAX_ITER)
                {
                    double newx = cx * cx - cy * cy + x;
                    cy = 2 * cx * cy + y;
                    cx = newx;
                    iter++;
                }
                data[(i - chunk_start) * WIDTH + j] = iter;
            }
        }
    }
    
    // send data to the next process and receive data from the previous process
    for (i = chunk_start; i < chunk_end; i++)
    {
        if (i == chunk_start && rank > 0)
        {
            MPI_Recv(data - WIDTH, WIDTH, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        if (i == chunk_end - 1 && rank < size - 1)
        {
            MPI_Send(data + (chunk_size - 1) * WIDTH, WIDTH, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank < size - 1)
        {
            MPI_Send(data + (chunk_size - 1) * WIDTH, WIDTH, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank > 0)
        {
            MPI_Recv(data - WIDTH, WIDTH, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
    }
    
    // gather the results from all processes onto the root process
    if (rank == 0)
    {
        // allocate memory for the final image
        int* image = (int*)malloc(HEIGHT * WIDTH * sizeof(int));
        
        // allocate memory for the counts and displacements arrays
        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (i = 0; i < size; i++)
        {
            counts[i] = chunk_size * WIDTH;
            if (i < remainder)
            {
                counts[i] += WIDTH;
            }
            displs[i] = i * chunk_size * WIDTH;
            if (i < remainder)
            {
                displs[i] += i;
            }
            else
            {
                displs[i] += remainder;
            }
        }
        
        // gather the data from all processes onto the root process
        MPI_Gatherv(data, chunk_size * WIDTH, MPI_INT, image, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        
        // print the image to the console
        for (i = 0; i < HEIGHT; i++)
        {
            for (j = 0; j < WIDTH; j++)
            {
                if (image[i * WIDTH + j] == MAX_ITER)
                {
                    printf("*");
                }
                else if (image[i * WIDTH + j] > MAX_ITER / 2)
                {
                    printf(".");
                }
                else if (image[i * WIDTH + j] > MAX_ITER / 4)
                {
                    printf(":");
                }
                else if (image[i * WIDTH + j] > MAX_ITER / 8)
                {
                    printf("-");
                }
                else
                {
                    printf(" ");
                }
            }
            printf("\n");
        }
        
        // free the memory
        free(image);
        free(counts);
        free(displs);
    }
    else
    {
        // send the data to the root process
        MPI_Gatherv(data, chunk_size * WIDTH, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // free the memory
    free(data);
    
    // finalize MPI
    MPI_Finalize();
    
    return 0;
}

void compute_mandelbrot(int start, int end, int* data)
{
    int i, j, iter;
    for (i = start; i < end; i++)
    {
        for (j = 0; j < WIDTH; j++)
        {
            double x = ((double)j - (double)WIDTH / 2.0) * 4.0 / (double)WIDTH;
            double y = ((double)i - (double)HEIGHT / 2.0) * 4.0 / (double)WIDTH;
            double cx = x;
            double cy = y;
            iter = 0;
            while (cx * cx + cy * cy < 4 && iter < MAX_ITER)
            {
                double newx = cx * cx - cy * cy + x;
                cy = 2 * cx * cy + y;
                cx = newx;
                iter++;
            }
            data[(i - start) * WIDTH + j] = iter;
        }
    }
}
