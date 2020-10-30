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

float   SimdMulSum( float *, float *, int );
void    NonSimdMul(    float *, float *,  float *, int );
float   NonSimdMulSum( float *, float *, int );

#endif		// SIMD_H
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

#ifndef ARRAYSIZE
#define ARRAYSIZE	32000000
#endif

float A[ARRAYSIZE];
float B[ARRAYSIZE];
float C[ARRAYSIZE];

int main( int argc, char *argv[ ] )
{
#ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
        return 1;
#endif
    
        double SimdMulSumPerf = 0.;
        double NoneSimdMulSumPerf = 0.;

        for( int t = 0; t < NUMTRIES; t++ )
        {   
                //  SSE
                double time0 = omp_get_wtime( );
                SimdMulSum(A, B, ARRAYSIZE);
                double time1 = omp_get_wtime( );
                double perf = (double)ARRAYSIZE/(time1-time0)/1000000.;
                if( perf > SimdMulSumPerf )
                {   
                    SimdMulSumPerf = perf;
                }
            
                //  Non SSE
                time0 = omp_get_wtime( );
                NonSimdMulSum(A, B, ARRAYSIZE);
                time1 = omp_get_wtime( );
                perf = (double)ARRAYSIZE/(time1-time0)/1000000.;
                if( perf > NoneSimdMulSumPerf )
                {
                    NoneSimdMulSumPerf = perf;
                }
        }

        printf( "  SimdMulSum Peak Performance = %8.4lf MegaMults/Sec\n", SimdMulSumPerf );
        printf( "  NoneSimdMulSum Peak Performance = %8.4lf MegaMults/Sec\n", NoneSimdMulSumPerf );
        printf( "  Speedup for SimdMulSum = %8.4f\n", SimdMulSumPerf / NoneSimdMulSumPerf );
        
    
    return 0;
}