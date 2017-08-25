#include "simd.p5.h"
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifndef SIMD_H
#define SIMD_H

// SSE stands for Streaming SIMD Extensions

#define SSE_WIDTH	4

#define ALIGNED		__attribute__((aligned(16)))


//void	SimdMul(float *, float *, float *, int);
float	SimdMulSum(float *, float *, int);


#endif		// SIMD_H

//void SimdMul(float *a, float *b, float *c, int len);
float SimdMulSum(float *a, float *b, int len);


#define SIZE			32768
#define NUMTRIES       	10

float Array[2 * SIZE];
float Sums[1 * SIZE];
double sumMegaCalcs, maxMegaCalcs;

int main()
{
	std::ofstream sums;
	sums.open("openMP_autocorrelation_sums.csv");


	for (int test = 0; test < NUMTRIES; ++test) {

		//"open signals file\n");
		FILE *fp = fopen("signals.txt", "r");
		if (fp == NULL)
		{
			fprintf(stderr, "Cannot open file 'signals.txt'\n");
			exit(1);
		}
		int Size;
		fscanf(fp, "%d", &Size);
		Size = SIZE;



		for (int i = 0; i < Size; i++)
		{
			fscanf(fp, "%f", &Array[i]);
			Array[i + Size] = Array[i];		// duplicate the array
		}

		double time0 = omp_get_wtime();
		//printf("set time\n");

		for (int shift = 0; shift < Size; shift++)
		{
			Sums[shift] = SimdMulSum(&Array[0], &Array[0 + shift], Size);
		}

		double time1 = omp_get_wtime();
		//printf("get time\n");
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


float SimdMulSum(float *a, float *b, int len)
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = (len / SSE_WIDTH) * SSE_WIDTH;

	__asm
	(
		".att_syntax\n\t"
		"movq    -40(%rbp), %rbx\n\t"		// a
		"movq    -48(%rbp), %rcx\n\t"		// b
		"leaq    -32(%rbp), %rdx\n\t"		// &sum[0]
		"movups	 (%rdx), %xmm2\n\t"		// 4 copies of 0. in xmm2
		);

	for (int i = 0; i < limit; i += SSE_WIDTH)
	{
		__asm
		(
			".att_syntax\n\t"
			"movups	(%rbx), %xmm0\n\t"	// load the first sse register
			"movups	(%rcx), %xmm1\n\t"	// load the second sse register
			"mulps	%xmm1, %xmm0\n\t"	// do the multiply
			"addps	%xmm0, %xmm2\n\t"	// do the add
			"addq $16, %rbx\n\t"
			"addq $16, %rcx\n\t"
			);
	}

	__asm
	(
		".att_syntax\n\t"
		"movups	 %xmm2, (%rdx)\n\t"	// copy the sums back to sum[ ]
		);

	for (int i = limit; i < len; i++)
	{
		sum[i - limit] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}
