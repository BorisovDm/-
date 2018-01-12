#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 131072
#define SELECTION 100
#define DOUBLE_SIZE 8

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    double starttime, endtime, sendTime;
    int count, rank, length;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double buf[SIZE];
    void *ptr = (void*)malloc(SIZE * 10000);
    MPI_Buffer_attach(ptr, 9900 * SIZE + 100 * MPI_BSEND_OVERHEAD);

    if (rank == 0){
		for(length = 0; length < SIZE; length++)
			buf[length] = length;
		printf("message length(byte), time(seconds)\n");
    }
    length = 1;
    while(length <= SIZE)
    {
		sendTime = 0;
		int i = 0;
		MPI_Barrier(MPI_COMM_WORLD);
		for(i = 1; i <= SELECTION; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0)
			{
				starttime = MPI_Wtime();

				//MPI_Send(buf, length, MPI_DOUBLE, 1, 21, MPI_COMM_WORLD);
				//MPI_Ssend(buf, length, MPI_DOUBLE, 1, 21, MPI_COMM_WORLD);
				MPI_Bsend(buf, length, MPI_DOUBLE, 1, 21, MPI_COMM_WORLD);

				MPI_Recv(buf, length, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				endtime = MPI_Wtime();
				sendTime += (endtime - starttime);
			}else
				if (rank == 1)
				{
					MPI_Recv(buf, length, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					//MPI_Send(buf, length, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
					//MPI_Ssend(buf, length, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
					MPI_Bsend(buf, length, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
				}
		}
		if (rank == 0)
		{
			sendTime /= (2 * SELECTION);
			printf("%d %.15f\n", length * DOUBLE_SIZE, sendTime);
		}
		if (length < 50) length += 6;
		else 
			if (length < 150) length += 30;
			else
				if (length < 1000) length += 100;
				else
	    			if (length < 7000) length += 800;
					else length += 10000;
    }
    MPI_Finalize();
    return 0;
}