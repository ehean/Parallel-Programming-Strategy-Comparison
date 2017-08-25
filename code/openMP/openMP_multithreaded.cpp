#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>

#define SIZE			32768
#define NUMTRIES       	10

float Array[2 * SIZE];
float Sums[1 * SIZE];
double sumMegaCalcs, maxMegaCalcs;

int main()
{

	#ifndef _OPENMP
		fprintf(stderr, "OpenMP is not supported here -- sorry.\n");
		return 1;
	#endif

	omp_set_num_threads(8);

	std::ofstream sums;
	sums.open("openMP_autocorrelation_sums.csv");
	

	
	for (int test = 0; test < NUMTRIES; ++test) {
		
		FILE *fp;
		errno_t err = fopen_s(&fp, "signals.txt", "r");
		if (fp == NULL)
		{
			fprintf(stderr, "Cannot open file 'signals.txt'\n");
			exit(1);
		}
		int Size;
		fscanf_s(fp, "%d", &Size);
		Size = SIZE;
	
	
		
		for (int i = 0; i < Size; i++)
		{
			fscanf_s(fp, "%f", &Array[i]);
			Array[i + Size] = Array[i];		// duplicate the array
		}

		double time0 = omp_get_wtime();

		#pragma omp parallel for //(+:sum)
		for (int shift = 0; shift < Size; shift++)
		{
			float sum = 0.;
			for (int i = 0; i < Size; i++)
			{
				sum += Array[i] * Array[i + shift];
			}
			Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
		}


		double time1 = omp_get_wtime();
		double megaCalcs = (double)SIZE*SIZE / (time1 - time0) / 1000000.;
		sumMegaCalcs += megaCalcs;
		if (megaCalcs > maxMegaCalcs)
			maxMegaCalcs = megaCalcs;

		fclose(fp);
	}

	for (int i = 0; i < SIZE; ++i) {
		sums << i << ", " << Sums[i] << '\n';
	}

	double avgMegaCalcs = sumMegaCalcs / (double)NUMTRIES;
	printf("   Peak Performance = %8.2lf MegaCalcs/Sec\n", maxMegaCalcs);
	printf("Average Performance = %8.2lf MegaCalcs/Sec\n", avgMegaCalcs);

}