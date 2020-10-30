#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "simd.p5.h"

#define SIZE   32768
float Array[2*SIZE];
float B[2*SIZE];
float  Sums[1*SIZE];

unsigned int seed = 0;  // a thread-private variable

float Ranf( unsigned int *seedp,  float low, float high ) {
    float r = (float) rand_r( seedp );              // 0 - RAND_MAX
    
    return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}

float x = Ranf( &seed, -1.f, 1.f );

int main( int argc, char *argv[ ] ) {
    #ifndef _OPENMP
        fprintf( stderr, "OpenMP is not available\n" );
        return 1;
    #endif
    
    FILE *fp = fopen( "signal.txt", "r" );
    if( fp == NULL )
    {
        fprintf( stderr, "Cannot open file 'signal.txt'\n" );
        exit( 1 );
    }
    int Size;
    fscanf( fp, "%d", &Size );
    Size = SIZE;
    float *A = new float[2*Size];
    float *B = new float[2*Size];
    for( int i = 0; i < Size; i++ )
    {
        fscanf( fp, "%f", &Array[i] );
        Array[i+Size] = Array[i];        // duplicate the array
        A[i] = Ranf(&seed,Array[i],Array[i]);
        A[i+Size] = (&seed, Array[i], Array[i]);
    }
    fclose( fp );
    double time0 = omp_get_wtime( );
    for( int shift = 0; shift < 513; shift++ )
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
    return 0;
}