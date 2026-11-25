
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>

#include <sys/time.h>

struct timeval timer;

void init_timing(struct timeval* tstart)
{
	gettimeofday(tstart, NULL);
}

float elapsed_time(struct timeval tstart)
{
	struct timeval tmp;
	long long diff;
	gettimeofday(&tmp, NULL);
	diff = tmp.tv_usec - tstart.tv_usec;
	diff += (tmp.tv_sec - tstart.tv_sec) * 1000000;
	return ((float)diff*1.0e-6);
}


void create_matrices(float** A, float** B, float** C, float** C_blas, int M, int N, int K)
{
    size_t i;

    *A = (float*) calloc(M*K,sizeof(float));
    *B = (float*) calloc(K*N,sizeof(float));
    *C = (float*) calloc(M*N,sizeof(float));
    *C_blas = (float*) calloc(M*N,sizeof(float));

    for(i = 0; i < M*K; i++)
        (*A)[i] = ((float)rand()/RAND_MAX-0.5)*0.1;

    for(i = 0; i < K*N; i++)
        (*B)[i] = ((float)rand()/RAND_MAX-0.5)*0.1;
}

void matmul_blas(const float* A, const float* B, float* C, int M, int N, int K)
{
    float alpha = 1.0f, beta = 0.0f;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, alpha, A, M, B, K, beta, C, M);
}


//Naive triple loop matmul
void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K)
{
	// Add your hown naive 3-loop implementation here
	int i;
	int j;
	int k;

	for(i=0; i<M; i ++)
		for(j=0; j<N; j ++)
			for(k=0; k<K; k ++)
				C[j*M+i] += A[k*M+i] * B[j*K+k];

}

void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K)
{
	// Add your hown naive 3-loop implementation here
	int i;
	int j;
	int k;

	for(i=0; i<M; i ++)
		for(j=0; j<N; j ++){
			float x=0.0f;	
			for(k=0; k<K; k ++)
				x += A[k*M+i] * B[j*K+k];
			C[j*M+i] = x;		
		}
}

void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K)
{
	// Add your hown naive 3-loop implementation here
	int i;
	int j;
	int k;
	
	float *A_T = malloc(K * M * sizeof(float));
	for(j=0; j<M; j ++)
		for(i=0; i<K; i ++){
			A_T[K*j+i] = A[M*i+j];
		}

	for(j=0; j<N; j ++)
		for(i=0; i<M; i ++){
			float x=0.0f;	
			for(k=0; k<K; k ++)
				x += A_T[i*K+k] * B[j*K+k];
			C[j*M+i] = x;		
		}
}


// Define a memory aligned vector type of 32 Bytes -> 8 float of 4 bytes (32 bits) each
typedef float vec __attribute__ (( vector_size(32) ));

void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K)
{
	if(K % 8 != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}
	
	// Some vector creation and usage exemples
	// Dynamically allocate a space in memory composed of X elements using the size of one element as alignement
	// vec *a = (vec*) aligned_alloc(sizeof(vec),X*sizeof(vec));  
	
	// Allocate and initialize a vector to a constant value
	// vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	
	// Fill a vector through indexing in a loop
	// for (i = 0; i < 8; i++)
	//     v[i] = i;
	
	// Access element i of a vectorized version of the matrix A
	// a[i/8][i%8] = A[i];
	
	
	
	//1° Create the vectorized versions of A and B

	vec *a = (vec*) aligned_alloc(sizeof(vec),(size_t)K*M*sizeof(float));
	vec *b = (vec*) aligned_alloc(sizeof(vec),(size_t)K*N*sizeof(float));

	//2° Fill the vectorized versions
	for(int i=0; i<M;i++)
		for (int k=0; k<K;k++){
			int idx = i*K+k;
			a[idx/8][idx%8]=A[k*M+i];
		}

	for(int j=0; j<N;j++)
		for (int k=0; k<K;k++){
			int idx = K*j+k;
			b[idx/8][idx%8]=B[idx];
		}
	
	int steps = K/8;
	
	//3° Do the i,j,k loops with the vectorized accumulate loop over k and fill the corresponding cell of C
	for(int j=0; j<N; j ++)
		for(int i=0; i<M; i ++){
			vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};	
			for(int k=0; k<steps; k ++)
				acc += a[i*steps+k] * b[j*steps+k];
			float sum = 0.0f;
			for(int l=0;l<8;l++)
				sum += acc[l];
			C[M*j+i] = sum;
		}

}



int main(int argc, char *argv[])
{
	size_t i;
	
	int l_size, M, N, K;
	float *A, *B, *C, *C_blas;

	if (argc == 2)
		l_size = atoi(argv[1]);
	else
		l_size = 512;
	
	M = l_size; N = l_size; K = l_size;

	create_matrices(&A, &B, &C, &C_blas, M, N, K);

	init_timing(&timer);

	matmul_blas(A,B,C_blas,M,N,K);

	printf("Regular sgemm time: %f s\n", elapsed_time(timer));

	init_timing(&timer);

	matmul_v4(A,B,C,M,N,K); //your custom matmul implementation
	
	printf("Custom sgemm time: %f s\n", elapsed_time(timer));
	
	//Verify the content of C against the OpenBLAS matmul
	for (i = 0; i < M*N; i++)
		if((C[i] - C_blas[i])*(C[i] - C_blas[i]) > 0.00001)
		{
			printf("ERROR ! MATRIX DIFF !\n");
			printf("i:%ld C:%f C_blas:%f\n", i, C[i], C_blas[i]);
			exit(EXIT_FAILURE);
		}
	exit(EXIT_SUCCESS);
}

		
		
