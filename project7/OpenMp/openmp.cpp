#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define NUMT	         1
#define ARRAYSIZE       10000	// you decide
#define NUMTRIES        100	// you decide

#define SIZE   32768
float Array[2*SIZE];
float  Sums[1*SIZE];

float A[ARRAYSIZE];
float B[ARRAYSIZE];
float C[ARRAYSIZE];

int
main( )
{

#ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
        return 1;
#endif

        omp_set_num_threads( NUMT );
        fprintf( stderr, "Using %d threads\n", NUMT );
    
        FILE *fp = fopen( "signal.txt", "r" );
        if( fp == NULL )
        {
            fprintf( stderr, "Cannot open file 'signal.txt'\n" );
        }
        int Size;
        fscanf( fp, "%d", &Size );
        Size = SIZE;
        float *Array = new float[ 2*Size ];
        float *Sums  = new float[ 1*Size ];
        for( int i = 0; i < Size; i++ )
        {
            fscanf( fp, "%f", &Array[i] );
            Array[i+Size] = Array[i];        // duplicate the array
        }
        fclose( fp );
        double time0 = omp_get_wtime( );
#pragma omp parallel for
        for( int shift = 0; shift < 512; shift++ )
        {
            float sum = 0.;
            for( int i = 0; i < Size; i++ )
            {
                sum += Array[i] * Array[i + shift];
            }
            Sums[shift] = sum;    // note the "fix #2" from false sharing if you are using OpenMP
            std::cout << " Shift: " << shift << " Sum: " << sum << "\n";
        }
        double time1 = omp_get_wtime( );
        std::cout << " GigaMultsPerSecond: " << (double)Size * (double)Size/(time1-time0)/1000000000 << "\n";
	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"

        return 0;
}