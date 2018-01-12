#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

void dec(const int N, const int rank, const int numtasks, int *begin, int *end)
{
    int step = N / numtasks;
    *begin = rank * step;
    *end = (rank + 1) * step;
    if(rank == numtasks - 1)
	*end = N;
}

//(a,b)
double dot(const double *a, const double *b, const int n)
{
    int i;
    double res = 0.0;
    for (i = 0; i < n; i++)
		res += a[i] * b[i];
    return res;
}

// c = alpha * a + beta * b
void addv(const double alpha, const double *a, const double beta, const double *b, const int n, double *c)
{
    int i;
    for (i = 0; i < n; i++)
		c[i] = alpha * a[i] + beta * b[i];
}

// b = a
void copyv(const double *a, const int n, double *b)
{
    memcpy(b, a, sizeof(double) * n);
}

/**
 * Read matrix from file:
 * M N
 * a11 a12 ..... a1N
 * a21 a22 ..... a2N
 * .....
 * aM1 ......... aMN
 */
void read_matrix(const char *path, double **A, int *m, int *n)
{
    int i, j;
    FILE *f = fopen(path, "r");
    fscanf(f, "%d", m);
    fscanf(f, "%d", n);
    double *t = (double*)malloc(sizeof(double) * (*m) * (*n));
    for (j = 0; j < *m; j++)
	{
		for (i = 0; i < *n; i++)
		{
			float fl;
			fscanf(f, "%lf", &fl); 
			t[i + (*n) * j] = fl;
		}
    }
    close(f);
    *A = t;
}

// d = A * x
void matvec(const double *A, const double *vector, const int n, double *b)
{
    int i, rank, numtasks, begin, end;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    dec(n, rank, numtasks, &begin, &end);

    double *x = (double*)malloc(n * sizeof(double));
    if(rank == 0) copyv(vector, n, x);

    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i < end - begin; i++)
		b[i] = dot(x, A + i * n, n);

    int *counts = (int*)malloc(numtasks * sizeof(int));
    int *displs = (int*)malloc(numtasks * sizeof(int));
    for(i = 0; i < numtasks; i++)
    {
		int left, right;
		dec(n, i, numtasks, &left, &right);
		counts[i] = (right - left);
		displs[i] = left;
    }
    MPI_Gatherv(b, end - begin, MPI_DOUBLE, b, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(counts);
    free(displs);
    free(x);
}

// x = A^-1 * b
void cgm(const double *A, const double *b, const int n, int *max_iter, double *tol, double *x)
{
    int k, rank, numtasks, begin, end;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    dec(n, rank, numtasks, &begin, &end);

    double *r, *p, *ap;
    double rr = 1;

    if(rank == 0)
    {
		r = (double*)malloc(sizeof(double) * n);
		p = (double*)malloc(sizeof(double) * n);
		ap = (double*)malloc(sizeof(double) * n);
    }
    else
    {
		r = (double*)malloc(sizeof(double) * (end - begin));
		ap = (double*)malloc(sizeof(double) * (end - begin));
    }

    matvec(A, x, n, r);//вектор х есть только у rank = 0
    if(rank == 0)
    {
		addv(1.0, b, -1.0, r, n, r);
		copyv(r, n, p);
		rr = dot(r, r, n);
    }

    for (k = 0; k < *max_iter; k++)
    {
		matvec(A, p, n, ap);//вектор p есть только у rank = 0
		double newrr;
		if(rank == 0)
		{
			double alpha = rr / dot(p, ap, n);
			addv(1.0, x, alpha, p, n, x);
			addv(1.0, r, -alpha, ap, n, r);
			newrr = dot(r, r, n);
		}

		MPI_Bcast(&newrr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (sqrt(newrr) < *tol)
		{
			rr = newrr;
			k++;
			break;
		}

		if(rank == 0)
		{
			addv(1.0, r, newrr / rr, p, n, p);
			rr = newrr;
		}
    }
    *max_iter = k;
    *tol = sqrt(rr);
    free(r);
    free(ap);
    if(rank == 0) free(p);
}


int main(int argc, char **argv)
{
    if(argc < 3)
    {
		printf("Usage: %s A b tolerance(default 1e-8) max_iter(default is b dimension)\n", argv[0]);
		exit(1);
    }

    MPI_Init(&argc, &argv);
    int rank, numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *A, *a, *b, *x;//A, b, x известны только rank = 0
    int N, i;
    double starttime;

    if(rank == 0)
    {
		starttime = MPI_Wtime();
		int M, N1, M1;
		read_matrix(argv[1], &A, &M, &N);
		if(M != N)
		{
			printf("Only square matrix are supported. Current matrix %dx%d.\n", M, N);
			exit(1);
		}
		read_matrix(argv[2], &b, &M1, &N1);
		if(M1 * N1 != N)
		{
			printf("Invalid b size %d.\n", M1 * N1);
			exit(1);
		}
		x = (double*)calloc(N, sizeof(double));
	}
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int begin, end;
    dec(N, rank, numtasks, &begin, &end);
    a = (double*)malloc((end - begin) * N * sizeof(double));

    int *sendcounts = (int*)malloc(numtasks * sizeof(int));
    int *displs = (int*)malloc(numtasks * sizeof(int));
    for(i = 0; i < numtasks; i++)
    {
		int left, right;
		dec(N, i, numtasks, &left, &right);
		sendcounts[i] = (right - left) * N;
		displs[i] = left * N;
    }
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, a, (end - begin) * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int max_iter = N;
    double tol = 1e-8;
    if (argc > 4) max_iter = atoi(argv[4]);
    if (argc > 3) tol = atof(argv[3]);

    cgm(a, b, N, &max_iter, &tol, x);

    double *t;
    if(rank == 0)
	{
		for (i = 0; i < N; i++) printf("%f ", x[i]);
		printf("\n");
		printf("tol %e\n", tol);
		printf("max_iter %d\n", max_iter);
		t = (double*)malloc(N * sizeof(double));
    }

    if(rank != 0) t = (double*)malloc((end - begin) * sizeof(double));
    matvec(a, x, N, t);
    if(rank == 0)
    {
		addv(1.0, t, -1.0, b, N, t);
		printf("||A*x-b|| %e\n", sqrt(dot(t, t, N)));
    }

    free(t);
    free(sendcounts);
    free(displs);
    free(a);
    if(rank == 0)
    {
		free(A);
		free(b);
		free(x);
		printf("\nWorking time(sec) = %.10f\n\n", MPI_Wtime() - starttime);
    }

    MPI_Finalize();
    return 0;
}