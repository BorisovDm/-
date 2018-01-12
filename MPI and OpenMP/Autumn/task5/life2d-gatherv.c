#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define ind(i, j) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx))
#define indp(i, j) (((i + l.nx) % l.nx) + ((j + l.ny) % l.ny) * (l.nx))
#define kp(k) ((k+l.p) % l.p)

typedef struct {
    int nx, ny;
    int *u0;
    int *u1;
    int steps;
    int save_steps;

    int ax, ay, bx, by; /* Зона ответственности k-го процесса, по осям x, y */
    int k; /* Номер текущего процесса. */
    int p; /* Число процессов. */
    int px; //кол-во процессов по х
    int py; //кол-во процессов по y
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);
void dec(const int k, const int nx, const int ny, const int px, const int py, int *ax, int *ay, int *bx, int *by);

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s input file.\n", argv[0]);
        return 0;
    }

    MPI_Init(&argc, &argv);
    life_t l;
    life_init(argv[1], &l);

    double starttime;
    if(l.k == 0){
	starttime = MPI_Wtime();
    }

    l.px = sqrt(l.p);
    while(l.p % l.px != 0){
        l.px--;
    }
    l.py = l.p / l.px;
    dec(l.k, l.nx, l.ny, l.px, l.py, &l.ax, &l.ay, &l.bx, &l.by);

    MPI_Datatype t_column;
    MPI_Type_vector(l.by - l.ay + 1, 1, l.nx, MPI_INT, &t_column);
    MPI_Type_commit(&t_column);

    MPI_Datatype t_block;
    MPI_Type_vector(l.by - l.ay + 1, l.bx - l.ax + 1, l.nx, MPI_INT, &t_block);
    MPI_Type_commit(&t_block);

    int i;
    for(i = 0; i < l.steps; i++) {
	if (i % l.save_steps == 0) {

	    int j, ax, ay, bx, by;
	    int *displs, *rcounts;
	    displs = (int*)malloc(l.p * sizeof(int));
	    rcounts = (int*)malloc(l.p * sizeof(int));
	    for (j = 0; j < l.p; j++) {
		if(j == 0){
		    displs[j] = 0;
		}
		else{
    		    displs[j] = displs[j - 1] + rcounts[j - 1];
		}
		dec(j, l.nx, l.ny, l.px, l.py, &ax, &ay, &bx, &by);
		rcounts[j] = (by - ay + 1) * (bx - ax + 1);
	    }
	    MPI_Gatherv(&l.u0[indp(l.ax, l.ay)], 1, t_block, l.u1, rcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

	    if(l.k == 0){
		int m, k;
		for(j = 0; j < l.p; j++){
		    dec(j, l.nx, l.ny, l.px, l.py, &ax, &ay, &bx, &by);
		    for(m = 0; m < by - ay + 1; m++){
			for(k = 0; k < bx - ax + 1; k++){
			    l.u0[indp(ax + k, ay + m)] = l.u1[displs[j] + k + m * (bx - ax + 1)];
			}
		    }
		}
	    	char buf[100];
		sprintf(buf, "life_%06d.vtk", i);
		printf("Saving step %d to '%s'.\n", i, buf);
		life_save_vtk(buf, &l);
	    }
	}
    //перессылка между процессами
//сообщения вверх-вниз
	MPI_Request rq[16];
	int addressee;

	MPI_Isend(&l.u0[indp(l.ax, l.ay)], l.bx - l.ax + 1, MPI_INT, kp(l.k - l.px), 0, MPI_COMM_WORLD, &rq[0]); //отправка вверх
	MPI_Irecv(&l.u0[indp(l.ax, l.by + 1)], l.bx - l.ax + 1, MPI_INT, kp(l.k + l.px), 0, MPI_COMM_WORLD, &rq[1]); //прием от низа

	MPI_Isend(&l.u0[indp(l.ax, l.by)], l.bx - l.ax + 1, MPI_INT, kp(l.k + l.px), 1, MPI_COMM_WORLD, &rq[2]); //отправка вниз
	MPI_Irecv(&l.u0[indp(l.ax, l.ay - 1)], l.bx - l.ax + 1, MPI_INT, kp(l.k - l.px), 1, MPI_COMM_WORLD, &rq[3]); //прием от верха
//сообщения по 1 элементу по диагоналям
//1 блок send-recv
	addressee = l.k - l.px - 1;
	if(l.ay == 0){addressee = l.k - 1 + l.p - l.px;}
	if(l.ax == 0){addressee = (l.k - 1 + l.p) % l.p;}
	MPI_Isend(&l.u0[indp(l.ax, l.ay)], 1, MPI_INT, addressee, 2, MPI_COMM_WORLD, &rq[4]); //отправка вверх-влево

	addressee = l.k + l.px + 1;
	if(l.by == l.ny - 1){addressee = l.k + l.px + 1 - l.p;}
	if(l.bx == l.nx - 1){addressee = (l.k + 1) % l.p;}
	MPI_Irecv(&l.u0[indp(l.bx + 1, l.by + 1)], 1, MPI_INT, addressee, 2, MPI_COMM_WORLD, &rq[5]); //прием от низ-право
//2 блок send-recv
	addressee = l.k - l.px + 1;
	if(l.ay == 0){addressee = l.k + 1 - l.px + l.p;}
	if(l.bx == l.nx - 1){addressee = (l.k + 1 - 2 * l.px + l.p) % l.p;}
    	MPI_Isend(&l.u0[indp(l.bx, l.ay)], 1, MPI_INT, addressee, 3, MPI_COMM_WORLD, &rq[6]); //отправка вверх-вправо

	addressee = l.k + l.px - 1;
	if(l.by == l.ny - 1){addressee = l.k - 1 + l.px - l.p;}
	if(l.ax == 0){addressee = (l.k - 1 + 2 * l.px + l.p) % l.p;}
	MPI_Irecv(&l.u0[indp(l.ax - 1, l.by + 1)], 1, MPI_INT, addressee, 3, MPI_COMM_WORLD, &rq[7]); //прием от низ-лево
//3 блок send-recv
	addressee = l.k - 1 + l.px;
	if(l.by == l.ny - 1){addressee = l.k - 1 + l.px - l.p;}
	if(l.ax == 0){addressee = (l.k - 1 + 2 * l.px) % l.p;}
	MPI_Isend(&l.u0[indp(l.ax, l.by)], 1, MPI_INT, addressee, 4, MPI_COMM_WORLD, &rq[8]); //отправка вниз-влево

	addressee = l.k - l.px + 1;
	if(l.ay == 0){addressee = l.k + 1 + l.p - l.px;}
	if(l.bx == l.nx - 1){addressee = (l.k + 1 - 2 * l.px + l.p) % l.p;}
	MPI_Irecv(&l.u0[indp(l.bx + 1, l.ay - 1)], 1, MPI_INT, addressee, 4, MPI_COMM_WORLD, &rq[9]); //прием от верх-право
//4 блок send-recv

	addressee = l.k + l.px + 1;
	if(l.by == l.ny - 1){addressee = l.k + 1 + l.px - l.p;}
	if(l.bx == l.nx - 1){addressee = (l.k + 1) % l.p;}
	MPI_Isend(&l.u0[indp(l.bx, l.by)], 1, MPI_INT, addressee, 5, MPI_COMM_WORLD, &rq[10]); //отправка вниз-вправо

	addressee = l.k - l.px - 1;
	if(l.ay == 0){addressee = l.k - 1 - l.px + l.p;}
	if(l.ax == 0){addressee = (l.k - 1 + l.p) % l.p;}
	MPI_Irecv(&l.u0[indp(l.ax - 1, l.ay - 1)], 1, MPI_INT, addressee, 5, MPI_COMM_WORLD, &rq[11]); //прием от верх-лево

//передача столбцов
        addressee = ((int)(l.k / l.px) == (int)((l.k + 1) / l.px)) ? l.k + 1 : l.k + 1 - l.px;
	MPI_Isend(&l.u0[indp(l.bx, l.ay)], 1, t_column, addressee, 6, MPI_COMM_WORLD, &rq[12]); //отправка правому
	if(l.k == 0){
	    addressee = l.px - 1;
	}else{
	    addressee = ((int)(l.k / l.px) == (int)((l.k - 1) / l.px)) ? l.k - 1 : l.k - 1 + l.px;
	}
    	MPI_Irecv(&l.u0[indp(l.ax - 1, l.ay)], 1, t_column, addressee, 6, MPI_COMM_WORLD, &rq[13]); //прием от левого

	if(l.k == 0){
	    addressee = l.px - 1;
	}else{
	    addressee = ((int)(l.k / l.px) == (int)((l.k - 1) / l.px)) ? l.k - 1 : l.k - 1 + l.px;
	}
	MPI_Isend(&l.u0[indp(l.ax, l.ay)], 1, t_column, addressee, 7, MPI_COMM_WORLD, &rq[14]); //отправка левому
	addressee = ((int)(l.k / l.px) == (int)((l.k + 1) / l.px)) ? l.k + 1 : l.k + 1 - l.px;
	MPI_Irecv(&l.u0[indp(l.bx + 1, l.ay)], 1, t_column, addressee, 7, MPI_COMM_WORLD, &rq[15]); //прием от правого

	MPI_Waitall(16, rq, MPI_STATUSES_IGNORE);
	life_step(&l);
    }
    MPI_Type_free(&t_column);
    MPI_Type_free(&t_block);
    life_free(&l);
    if(l.k == 0){
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
    MPI_Comm_size(MPI_COMM_WORLD, &l->p);
    MPI_Comm_rank(MPI_COMM_WORLD, &l->k);

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
    if(l->k == 0){
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

void life_step(life_t *l)
{
    int i, j;
    for (j = l->ay; j <= l->by; j++) {
	for (i = l->ax; i <= l->bx; i++) {
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
void dec(const int k, const int nx, const int ny, const int px, const int py, int *ax, int *ay, int *bx, int *by)
{
    int column = k % px;
    int row = k / px;
    int x_step = nx / px;
    int y_step = ny / py;

    *ax = column * x_step;
    *bx = *ax + x_step - 1;
    if(column == px - 1){
	*bx = nx - 1;
    }
    *ay = row * y_step;
    *by = *ay + y_step - 1;
    if(row == py - 1){
	*by = ny - 1;
    }
}
