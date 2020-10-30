#include <xmmintrin.h>
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
float
SimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps( &sum[0] );
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for( int i = limit; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}

void
NonSimdMul( float *a, float *b,   float *c,   int len )
{
	for( int i = 0; i < len; i++ )
	{
		c[i] = a[i] * b[i];
	}
}

float
NonSimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;

	for( int i = 0; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0];
}