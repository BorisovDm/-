#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int power2(int i)
{
	return 1 << i;
}

void merge(int *a, int *b, int *c, int na, int nb);
void merge_sort(int *a, int na);
void print(int *a, int na);
int check_sort(int *a, int n);
double timer();
int n_count(int n, int rank, int size);

int main(int argc, char *argv[])
{
	int i, n, nlocal, npartner, *a, *b, *c;
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double t;
	if (argc < 2)
	{
		if (rank == 0) printf("Usage: %s num_elements.\n", argv[0]);
		return 1;
	}
	n = atoi(argv[1]);
	nlocal = n_count(n, rank, size);
	a = (int*)malloc(sizeof(int) * nlocal);
	
	srand(time(NULL));
	for (i = 0; i < nlocal; i++) a[i] = rand() % 100;
	
	t = timer();
	merge_sort(a, nlocal);

	/* Начало сбора. */
	int d = ceil(log2(size));
	int mask = 0;
	for(i = 0; i < d; i++)
	{
	    if ((rank & mask) == 0)
	    {
			int partner = rank ^ power2(i);
			if(partner < size)
			{
				if((rank & power2(i)) != 0)
				{
					MPI_Send(&nlocal, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
					MPI_Send(a, nlocal, MPI_INT, partner, 0, MPI_COMM_WORLD);
					printf("step = %d, %d send to %d\n", i, rank, partner);
				}
				else
				{
					int n_partner;
					MPI_Recv(&n_partner, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					b = (int*)malloc(sizeof(int) * n_partner);
					MPI_Recv(b, n_partner, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					c = (int*)malloc(sizeof(int) * (n_partner + nlocal));
					merge(a, b, c, nlocal, n_partner);
					free(a);
					a = c;
					nlocal += n_partner;
					free(b);
				}
			}
	    }
	    mask = mask ^ power2(i);
	}
	/* Конец сбора. */

	if (rank == 0)
	{
		t = timer() - t;
		if (n < 11) print(a, n);
		printf("Time: %f sec, sorted: %d\n", t, check_sort(a, n));
		printf("Time(sec):\n%f\n", t);
	}
	free(a);
	MPI_Finalize();
	return 0;
}

int n_count(int n, int rank, int size)
{
	return (n / size) + (n % size > rank);
}

double timer()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + 1e-6 * (double)ts.tv_usec;
}

int check_sort(int *a, int n)
{
	int i;
	for (i = 0; i < n - 1; i++)
		if (a[i] > a[i+1]) return 0;
	return 1;
}

void print(int *a, int na)
{
	int i;
	for (i = 0; i < na; i++) printf("%d ", a[i]);
	printf("\n");
}

/*
 * Процедура слияния массивов a и b в массив c.
 */
void merge(int *a, int *b, int *c, int na, int nb)
{
	int i = 0, j = 0;
	while (i < na && j < nb)
	{
		if (a[i] <= b[j])
		{
			c[i + j] = a[i];
			i++;
		}
		else
		{
			c[i + j] = b[j];
			j++;
		}
	}
	if (i < na) memcpy(c + i + j, a + i, (na - i) * sizeof(int));
	else memcpy(c + i + j, b + j, (nb - j) * sizeof(int));
}

/*
 * Процедура сортировки слиянием.
 */
void merge_sort(int *a, int na)
{
	if(na < 2) return;
	if(na == 2)
	{
		if(a[0] > a[1])
		{
			int t = a[0];
			a[0] = a[1];
			a[1] = t;
		}
		return;
	}
	merge_sort(a, na / 2);
	merge_sort(a + na / 2, na - na / 2);

	int *b = (int*)malloc(sizeof(int) * na);
	
	merge(a, a + na / 2, b, na / 2, na - na / 2);
	
	memcpy(a, b, sizeof(int) * na);
	
	free(b);
}