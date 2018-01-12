#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void merge(int *a, int *b, int *c, int na, int nb);
void merge_sort(int *a, int na);
void parallel_sort(int *a, int na, int nt);
void print(int *a, int na);
int check_sort(int *a, int n);
double timer();
int n_start(int i, int n);

int pow2(int i)
{
    return 1 << i;
}

int main (int argc, char *argv[])
{
	int i, n, *a, *b;
	double t;
	if (argc < 2)
	{
		printf("Usage: %s num_elements.\n", argv[0]);
		return 1;
	}
	n = atoi(argv[1]);
	a = (int*)malloc(sizeof(int) * n);
	srand(time(NULL));

	for (i = 0; i < n; i++) a[i] = rand() % 1000;
	if (n < 101) print(a, n);

	t = timer();

	int size = omp_get_max_threads();
	b = (int*)malloc(sizeof(int) * size * 2);
	int mask = 0, d = ceil(log2(size));

	#pragma omp parallel private(i)
	{
	    int rank = omp_get_thread_num();
	    b[rank * 2] = n_start(rank, n);
	    b[rank * 2 + 1] = n_start(rank + 1, n);
	    merge_sort(a + b[rank * 2], b[rank * 2 + 1] - b[rank * 2]);

	    #pragma omp barrier
	    for(i = 0; i < d; i++)
	    {
			if((rank & mask) == 0)
			{
				int partner = rank ^ pow2(i);
				if(partner < size)
				{
					if((rank & pow2(i)) == 0)
					{
						int *c = (int*)malloc(sizeof(int) * (b[partner * 2 + 1] - b[rank * 2]));
						merge(a + b[rank * 2], a + b[partner * 2], c, b[rank * 2 + 1] - b[rank * 2], b[partner * 2 + 1] - b[partner * 2]);
						memcpy(a + b[rank * 2], c, sizeof(int) * (b[partner * 2 + 1] - b[rank * 2]));
						free(c);

						b[rank * 2 + 1] = b[partner * 2 + 1];
					}
				}
			}
			#pragma omp barrier
			#pragma omp single
			mask = mask ^ pow2(i);
	    }
	}

	t = timer() - t;
	if (n < 101) print(a, n);
	printf("Time: %f sec\n", t);
	free(b);
	free(a);
	return 0;
}

int n_start(int i, int n)
{
    int threads = omp_get_max_threads();
    int step = n / threads;
    int start = step * i;
    if (i == threads) start = n;
    return start;
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
	merge_sort(a, na / 2);
	merge_sort(a + na / 2, na - na / 2);
	int *b = (int*)malloc(sizeof(int) * na);
	merge(a, a + na / 2, b, na / 2, na - na / 2);
	memcpy(a, b, sizeof(int) * na);
	free(b);
}