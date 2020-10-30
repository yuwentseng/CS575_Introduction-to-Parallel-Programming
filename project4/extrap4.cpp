#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include "extrasimd.h"
#include <time.h>

// setting the number of nodes
#ifndef ARRAYSIZE
#define ARRAYSIZE	8000000
#endif

#ifndef NUMTRIES
#define NUMTRIES	25
#endif

//NUMT
#ifndef NUMT
#define NUMT		16
#endif 

#define NUM_ELEMENTS_PER_CORE ARRAYSIZE / NUMT

float a[ARRAYSIZE];
float b[ARRAYSIZE];
float c[ARRAYSIZE];

float
Ranf(float low, float high)
{
	float r = (float)rand();               // 0 - RAND_MAX
	float t = r / (float)RAND_MAX;       // 0. - 1.

	return   low + t * (high - low);
}

void fillArry(float* arr, int len)
{
	for (int i = 0; i < len; i++)
	{
		arr[i] = Ranf(-1, 1);
	}
}

void
TimeOfDaySeed()
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time(&timer);
	double seconds = difftime(timer, mktime(&y2k));
	unsigned int seed = (unsigned int)(1000.*seconds);    // milliseconds
	srand(seed);
}

int main() 
{
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif

	// setting for omp-SimdMulSum
	double ompmegaCalcsPerSecond;
	double sum_ompmegaCalcsPerSecond = 0;
	double max_ompmegaCalcsPerSecond = 0;

	fillArry(a, ARRAYSIZE);
	fillArry(b, ARRAYSIZE);
	//fillArry(c, ARRAYSIZE);

	TimeOfDaySeed();

	for (int t = 0; t < NUMTRIES; t++)
	{
		double time4 = omp_get_wtime();
		#pragma omp parallel
		{
			int first = omp_get_thread_num() * NUM_ELEMENTS_PER_CORE;
			SimdMulSum(a, b, NUM_ELEMENTS_PER_CORE);
		}

		double time5 = omp_get_wtime();

		ompmegaCalcsPerSecond = (double)(ARRAYSIZE) / (time5 - time4) / 1000000.;

		if (ompmegaCalcsPerSecond > max_ompmegaCalcsPerSecond) {
			max_ompmegaCalcsPerSecond = ompmegaCalcsPerSecond;
		}
	}

	printf("Peak Performance of ompsimd = \t%8.13lf\n", max_ompmegaCalcsPerSecond);
}


