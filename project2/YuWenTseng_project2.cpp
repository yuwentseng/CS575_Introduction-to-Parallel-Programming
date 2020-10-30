#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>

// ranges for the random numbers:
#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.

// setting the number of threads:
#ifndef NUMT
#define NUMT		1 	//2 ,4, 8
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	5 	//10, 100, 500, 1000, 1500, 2000, 2500, 3000
#endif

// a variety of number of subdivisions
#ifndef NUMNODES
#define NUMNODES  5 	//10, 100, 500, 1000, 1500, 2000, 2500, 3000
#endif

// use parallel reduction to compute the volume of a superquadric using N=4
#ifndef N
#define N 4
#endif

//float Height( int, int );

float
Height(int iu, int iv)	// iu,iv = 0 .. NUMNODES-1
{
	float x = -1. + 2.*(float)iu / (float)(NUMNODES - 1);	// -1. to +1.
	float y = -1. + 2.*(float)iv / (float)(NUMNODES - 1);	// -1. to +1.

	float xn = pow(fabs(x), (double)N);
	float yn = pow(fabs(y), (double)N);
	float r = 1. - xn - yn;
	if (r < 0.)
		return 0.;
	float height = pow(1. - xn - yn, 1. / (float)N);
	return height;
}

int main(int argc, char *argv[])
{
	#ifndef _OPENMP
		fprintf(stderr, "OpenMp is not available!\n");
		return 1;
	#endif

	int numprocs = omp_get_num_procs();
	omp_set_num_threads(NUMT);	

	double maxHeights = 0.;
	//double sumHeights = 0.;
	// the area of a single full-sized tile:
	float fullTileArea = (((XMAX - XMIN) / (double)(NUMNODES - 1)) * ((YMAX - YMIN) / (double)(NUMNODES - 1)));
	double volume = 0.;

	for (int i = 0; i < NUMTRIES; i++)
	{		
		double time0 = omp_get_wtime();

		// sum up the weighted heights into the variable "volume"
		// using an OpenMP for loop and a reduction:
		#pragma omp parallel for default(none),shared(fullTileArea),reduction(+:volume)
		for (int i = 0; i < NUMNODES * NUMNODES; i++) {
			
			int iu = i % NUMNODES;
			int iv = i / NUMNODES;
			double z = Height(iu, iv);
			
			//Adding 
			bool _iu_edge = iu == 0 || iu == NUMNODES - 1;  
			bool _iv_edge = iv == 0 || iv == NUMNODES - 1;  

			if (_iu_edge && _iv_edge){
				volume = volume + z * (fullTileArea / 4.);
			}
			else if(_iu_edge){             
				volume = volume + z * (fullTileArea / 2.);
			}
			else if(_iv_edge){             
				volume = volume + z * (fullTileArea / 2.);
			}
			else {
				volume = volume + z * fullTileArea * 2;
			}
		}
		double time1 = omp_get_wtime();

		double megaHeights = (double)(NUMNODES*NUMNODES) / (time1 - time0) / 1000000.;
		//sumHeights = sumHeights + megaHeights;

		if (megaHeights >= maxHeights){
			maxHeights = megaHeights;
		}
		else if(megaHeights < maxHeights)
		{
			maxHeights = maxHeights;
		}
		printf("volume is %f\n", volume);
	}
	
	fprintf(stderr, "   Peak Performance =\t%8.3lf MegaHeights/Sec\n", maxHeights);
	//double avgMegaHeights = sumHeights / (double)NUMTRIES;
	//fprintf(stderr, "Average Performance =\t%8.3lf MegaHeights/Sec\n\n", avgMegaHeights);
	return 0;

}