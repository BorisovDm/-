#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>

#define ind(i, j) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx))
#define ind2(i, j) (((i + l.nx) % l.nx) + ((j + l.ny) % l.ny) * (l.nx))

typedef struct {
    int nx, ny;
    int *u0;
    int *u1;
    int steps;
    int save_steps;
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l, int start, int end);
void life_save_vtk(const char *path, life_t *l);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int numtasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double starttime;
    if(rank == 0){
	starttime = MPI_Wtime();
    }
    if (argc != 2) {
        printf("Usage: %s input file.\n", argv[0]);
        return 0;
    }
    life_t l;
    life_init(argv[1], &l);

    int *seg; //segmentation
    seg = (int*)malloc(numtasks * 2 * sizeof(int)); //start & end
    int size = l.ny, number = numtasks;
    int i = 0, j = 0, task, current = 0;

    for(i = 0; i < numtasks; i++){
        task = (int)(size / number);
		seg[i * 2] = current;
		seg[i * 2 + 1] = seg[i * 2] + task - 1;
		current = seg[2 * i + 1] + 1;
		size -= task;
		number--;
    }

    for (i = 0; i < l.steps; i++) {
	if (i % l.save_steps == 0) {
	    //пересылка данных от всех к rank = 0
	    if(rank == 0){
			for(j = 1; j < numtasks; j++){
				MPI_Recv(&(l.u0[ind2(0, seg[j * 2])]), (seg[j * 2 + 1] - seg[j * 2] + 1) * l.nx, MPI_INT, j, 70, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			char buf[100];
			sprintf(buf, "life_%06d.vtk", i);
			printf("Saving step %d to '%s'.\n", i, buf);
			life_save_vtk(buf, &l);
	    }
	    else{
			MPI_Send(&(l.u0[ind2(0, seg[rank * 2])]), (seg[rank * 2 + 1] - seg[rank * 2] + 1) * l.nx, MPI_INT, 0, 70, MPI_COMM_WORLD);
	    }
	}
	//перессылка между процессами
	if(numtasks != 1){
	    if(rank % 2 == 0)
	    {
		if((numtasks % 2 == 1) && (rank == numtasks - 1)){
		    MPI_Recv(&(l.u0[ind2(0, seg[rank * 2] - 1)]), l.nx, MPI_INT, rank - 1, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //i-1 -> i
		    MPI_Send(&(l.u0[ind2(0, seg[rank * 2])]), l.nx, MPI_INT, rank - 1, 40, MPI_COMM_WORLD); //i -> i-1
		}
		else{
		    if((numtasks % 2 == 1) && (rank == 0)){
			MPI_Send(&(l.u0[ind2(0, seg[rank * 2 + 1])]), l.nx, MPI_INT, rank + 1, 10, MPI_COMM_WORLD); //i -> i+1
			MPI_Recv(&(l.u0[ind2(0, seg[rank * 2 + 1] + 1)]), l.nx, MPI_INT, rank + 1, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //i+1 -> i
		    }else{
			MPI_Send(&(l.u0[ind2(0, seg[rank * 2 + 1])]), l.nx, MPI_INT, rank + 1, 10, MPI_COMM_WORLD); //i -> i+1
			MPI_Recv(&(l.u0[ind2(0, seg[rank * 2 + 1] + 1)]), l.nx, MPI_INT, rank + 1, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //i+1 -> i
			MPI_Recv(&(l.u0[ind2(0, seg[rank * 2] - 1)]), l.nx, MPI_INT, (rank - 1 + numtasks) % numtasks, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //i-1 -> i
			MPI_Send(&(l.u0[ind2(0, seg[rank * 2])]), l.nx, MPI_INT, (rank - 1 + numtasks) % numtasks, 40, MPI_COMM_WORLD); //i -> i-1
		    }
		}
	    }
	    if(rank % 2 != 0)
	    {
		MPI_Recv(&(l.u0[ind2(0, seg[rank * 2] - 1)]), l.nx, MPI_INT, rank - 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //i-1 -> i
		MPI_Send(&(l.u0[ind2(0, seg[rank * 2])]), l.nx, MPI_INT, rank - 1, 20, MPI_COMM_WORLD); //i -> i-1
		MPI_Send(&(l.u0[ind2(0, seg[rank * 2 + 1])]), l.nx, MPI_INT, (rank + 1) % numtasks, 30, MPI_COMM_WORLD); //i -> (i+1)%numtasks
		MPI_Recv(&(l.u0[ind2(0, seg[rank * 2 + 1] + 1)]), l.nx, MPI_INT, (rank + 1) % numtasks, 40, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //i+1 -> i
	    }
	    if(numtasks % 2 == 1){
		if(rank == 0){
		    MPI_Send(&(l.u0[ind2(0, seg[rank * 2])]), l.nx, MPI_INT, numtasks - 1, 50, MPI_COMM_WORLD); //0 -> numtasks-1
		    MPI_Recv(&(l.u0[ind2(0, seg[rank * 2] - 1)]), l.nx, MPI_INT, numtasks - 1, 60, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //numtasks-1 -> 0
		}
		if(rank == numtasks - 1){
		    MPI_Recv(&(l.u0[ind2(0, seg[rank * 2 + 1] + 1)]), l.nx, MPI_INT, 0, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //0 -> numtasks-1
		    MPI_Send(&(l.u0[ind2(0, seg[rank * 2 + 1])]), l.nx, MPI_INT, 0, 60, MPI_COMM_WORLD); //numtasks-1 -> 0
		}
	    }
	}
	life_step(&l, seg[rank * 2], seg[rank * 2 + 1]);
    }

    life_free(&l);
    if(rank == 0){
	printf("working time(sec) = %.10f\n", MPI_Wtime() - starttime);
    }
    MPI_Finalize();
    return 0;
}

/**
 * Загрузить входную конфигурацию.
 * Формат файла, число шагов, как часто сохранять, размер поля, затем идут координаты заполненых клеток:
 * steps
 * save_steps
 * nx ny
 * i1 j2
 * i2 j2
 */
void life_init(const char *path, life_t *l)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FILE *fd = fopen(path, "r");
    assert(fd);
    assert(fscanf(fd, "%d\n", &l->steps));
    assert(fscanf(fd, "%d\n", &l->save_steps));
    assert(fscanf(fd, "%d %d\n", &l->nx, &l->ny));

    l->u0 = (int*)calloc(l->nx * l->ny, sizeof(int));
    l->u1 = (int*)calloc(l->nx * l->ny, sizeof(int));

    int i, j, r, cnt = 0;
    while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
        l->u0[ind(i, j)] = 1;
	cnt++;
    }
    if(rank == 0){
	printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
	printf("Field size: %dx%d\n", l->nx, l->ny);
	printf("Loaded %d life cells.\n", cnt);
    }
    fclose(fd);
}

void life_free(life_t *l)
{
    free(l->u0);
    free(l->u1);
    l->nx = l->ny = 0;
}

void life_save_vtk(const char *path, life_t *l)
{
    FILE *f;
    int i1, i2, j;
    f = fopen(path, "w");
    assert(f);
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "Created by write_to_vtk2d\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");
    fprintf(f, "DIMENSIONS %d %d 1\n", l->nx+1, l->ny+1);
    fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
    fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
    fprintf(f, "CELL_DATA %d\n", l->nx * l->ny);

    fprintf(f, "SCALARS life int 1\n");
    fprintf(f, "LOOKUP_TABLE life_table\n");
    for (i2 = 0; i2 < l->ny; i2++){
    	for (i1 = 0; i1 < l->nx; i1++){
	    fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
	}
    }
    fclose(f);
}

void life_step(life_t *l, int start, int end)
{
    int i, j;
    for (j = start; j <= end; j++) {
	for (i = 0; i < l->nx; i++) {
	    int n = 0;
	    n += l->u0[ind(i+1, j)];
	    n += l->u0[ind(i+1, j+1)];
	    n += l->u0[ind(i,   j+1)];
	    n += l->u0[ind(i-1, j)];
	    n += l->u0[ind(i-1, j-1)];
	    n += l->u0[ind(i,   j-1)];
	    n += l->u0[ind(i-1, j+1)];
	    n += l->u0[ind(i+1, j-1)];
	    l->u1[ind(i,j)] = 0;
	    if (n == 3 && l->u0[ind(i,j)] == 0) {
		l->u1[ind(i,j)] = 1;
	    }
	    if ((n == 3 || n == 2) && l->u0[ind(i,j)] == 1) {
		l->u1[ind(i,j)] = 1;
	    }
	}
    }
    int *tmp;
    tmp = l->u0;
    l->u0 = l->u1;
    l->u1 = tmp;
}