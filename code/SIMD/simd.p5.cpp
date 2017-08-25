#include "simd.p5.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>

#include <omp.h>

#ifndef SIMD_H
#define SIMD_H

// SSE stands for Streaming SIMD Extensions

#define SSE_WIDTH	4

#define ALIGNED		__attribute__((aligned(16)))


void	SimdMul(float *, float *, float *, int);
float	SimdMulSum(float *, float *, int);


#endif		// SIMD_H

#define NUMTRIES       	10 

void SimdMul(float *a, float *b, float *c, int len);
float SimdMulSum(float *a, float *b, int len);
float Ranf(float low, float high);


int main()
{

	const int ARRSIZECOUNT = 16;

	long int arrSize[ARRSIZECOUNT] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };

	double maxmegaMults = 0.;
	double sumMegaMults = 0.;
	double sumTime = 0.;
	double elapsed;

	// create csv output files to store performance results
	std::ofstream peak, avg;
	peak.open("peak_performance_mults.csv");
	avg.open("avg_performance_mults.csv");

	// add column headers to csv files

	for (int idx = 0; idx < ARRSIZECOUNT; idx++) {
		peak << "," << arrSize[idx];
		avg << "," << arrSize[idx];
	}

	avg << "\n";
	peak << "\n";

	for (int i = 0; i < ARRSIZECOUNT; ++i) {
		arrSize[i] *= 1000;
	}

	// iterate through each array size
	for (int sizeIdx = 0; sizeIdx < ARRSIZECOUNT; sizeIdx++) {

		// Initialize arrays
		float *A = new float[arrSize[sizeIdx]];
		float *B = new float[arrSize[sizeIdx]];
		float *C = new float[arrSize[sizeIdx]];

		for (int i = 0; i < arrSize[sizeIdx]; ++i) {
			A[i] = Ranf(-1.f, 1.f);
			B[i] = Ranf(-1.f, 1.f);
		}

		maxmegaMults = 0;
		sumMegaMults = 0;

		// iterate through each try
		for (int t = 0; t < NUMTRIES; t++)
		{

			// set timer
			double time0 = omp_get_wtime();

			SimdMul(A, B, C, arrSize[sizeIdx]);

			// stop timer
			double time1 = omp_get_wtime();
			double megaMults = (double)(arrSize[sizeIdx]) / (time1 - time0) / 1000000.;
			elapsed = (time1 - time0) / 1000000.;
			sumMegaMults += megaMults;

			if (megaMults > maxmegaMults)
				maxmegaMults = megaMults;
		}

		double avgmegaMults = sumMegaMults / (double)NUMTRIES;

		peak << "," << maxmegaMults;
		avg << "," << avgmegaMults;

		// print results to console
		printf("Array Size: %d\n", arrSize[sizeIdx]);
		printf("   Peak Performance = %8.2lf megaMults/Sec\n", maxmegaMults);
		printf("Average Performance = %8.2lf megaMults/Sec\n", avgmegaMults);
		printf(" elapsed = %8.2lf \n\n", elapsed);


		delete A;
		delete B;
		delete C;

	}

	peak << "\n";
	avg << "\n";
	

	peak.close();
	avg.close();

	return 0;
}


float Ranf(float low, float high)
{
	float r = (float)rand();		// 0 - RAND_MAX

	return(low + r * (high - low) / (float)RAND_MAX);
}


void SimdMul(float *a, float *b, float *c, int len)
{
	int limit = (len / SSE_WIDTH) * SSE_WIDTH;
	__asm
	(
		".att_syntax\n\t"
		"movq    -24(%rbp), %rbx\n\t"		// a
		"movq    -32(%rbp), %rcx\n\t"		// b
		"movq    -40(%rbp), %rdx\n\t"		// c
		);

	for (int i = 0; i < limit; i += SSE_WIDTH)
	{
		__asm
		(
			".att_syntax\n\t"
			"movups	(%rbx), %xmm0\n\t"	// load the first sse register
			"movups	(%rcx), %xmm1\n\t"	// load the second sse register
			"mulps	%xmm1, %xmm0\n\t"	// do the multiply
			"movups	%xmm0, (%rdx)\n\t"	// store the result
			"addq $16, %rbx\n\t"
			"addq $16, %rcx\n\t"
			"addq $16, %rdx\n\t"
			);
	}

	for (int i = limit; i < len; i++)
	{
		c[i] = a[i] * b[i];
	}
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
