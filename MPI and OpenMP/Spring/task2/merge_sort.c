#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void merge(int *a, int *b, int *c, int na, int nb);
void merge_sort(int *a, int na);
void print(int *a, int na);
int check_sort(int *a, int n);
double timer();

int main (int argc, char *argv[])
{
    int i, n, *a;
    double t;
    if (argc < 2)
	{
		printf("Usage: %s num_elements.\n", argv[0]);
		return 1;
    }
    n = atoi(argv[1]);
    a = (int*)malloc(sizeof(int) * n);
    srand(time(NULL));

    for (i = 0; i < n; i++) a[i] = rand() % 100;
    //if (n < 101) print(a, n);

    t = timer();
    merge_sort(a, n);
    t = timer() - t;

    //if (n < 101) print(a, n);
    //printf("Time: %f sec, sorted: %d\n", t, check_sort(a, n));
    printf("Time(sec):\n %f \n", t);
    free(a);
    return 0;
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
	while(i < na && j < nb)
	{
	    if (a[i] < b[j])
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
	if (i == na)
	{
	    while(j < nb)
		{
			c[i + j] = b[j];
			j++;
	    }
	}
	if (j == nb)
	{
	    while(i < na)
		{
			c[i + j] = a[i];
			i++;
	    }
	}
}

/*
 * Процедура сортировки слиянием.
 */
void merge_sort(int *a, int na)
{
    if(na < 2) return;
    else
    {
		int size1 = na / 2;
		int size2 = na - size1;
		merge_sort(a, size1);
		merge_sort(a + size1, size2);
		int *B = (int*)malloc(sizeof(int) * na);
		merge(a, a + size1, B, size1, size2);
		int element;
		for(element = 0; element < na; element++)
			a[element] = B[element];
		free(B);
    }
}