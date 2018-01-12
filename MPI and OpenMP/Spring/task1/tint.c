#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define A 2.0

double f(double x)
{
    return sqrt(4.0 - x * x);
}

int power2(int d)
{
    return 1 << d;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
		printf("Usage: %s num_intervals.\n", argv[0]);
		return 1;
    }
    MPI_Init(&argc, &argv);
    int i, rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);
    double h = A / N;
    double S = 0.0;

    for (i = rank; i < N; i += size)
    {
	S += h * (f(h * i) + f(h * (i + 1))) / 2.0;
    }
    int d = ceil(log2(size));
    int mask = 0;
    for(i = 0; i < d; i++)
    {
		if ((rank & mask) == 0)
		{
			int partner = rank ^ power2(i);
			if(partner < size)
			{
				if ((rank & power2(i)) != 0)
					MPI_Send(&S, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD);
				else
				{
					double recv;
					MPI_Recv(&recv, 1, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					S += recv;
				}
			}
		}
        mask = mask ^ power2(i);
    }

    if (rank == 0) printf("%f\n", S);
    MPI_Finalize();
    return 0;
}