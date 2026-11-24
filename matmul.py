
import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt 

def matmul_naive(A, B, C, M, N, K):
	# Add your own naive 3-loop implementation here
	for i in range(M):
		for j in range(N):
			for k in range(K):
				C[i, j] += A[i][k]*B[k][j]
	return C

def matmul_numpy(A, B, C, M, N, K):
	for i in range(M):
		for j in range(N):
			C[i,j] = np.sum(A[i,:]*B[:,j])
	return C

@jit(nopython=True, cache=True, fastmath=False)
def matmul_compiled_naive(A, B, C, M, N, K):
	for i in range(M):
		for j in range(N):
			for k in range(K):
				C[i, j] += A[i][k]*B[k][j]
	return C

@jit(nopython=True, cache=True, fastmath=False)
def matmul_compiled_numpy(A, B, C, M, N, K):
	for i in range(M):
		for j in range(N):
			C[i,j] = np.sum(A[i,:]*B[:,j])
	return C


m_size = 512
M = m_size
N = m_size
K = m_size

np.random.seed(0)

A = ((np.random.random((M,K))-0.5)*0.1).astype("float32")
B = ((np.random.random((K,N))-0.5)-0.1).astype("float32")
C_verif = A@B
C = np.zeros((M,N), dtype="float32")

t_start = time.time()

# matmul_compiled(A, B, C, M, N, K)

# ellapsed_time = time.time() - t_start # In seconds, naive gets 53.76 seconds, numpy gets 0.73 seconds, compiled gets 0.65 seconds

# print(ellapsed_time)

# if(np.sum((C - C_verif)**2) > 0.00001):
# 	print("Error: results do not match!")
# 	exit()


"""Plots for question 6"""

funcs = [matmul_naive, matmul_numpy, matmul_compiled_naive, matmul_compiled_numpy]
funcs_d = {0: "matmul_naive",
           1: "matmul_numpy",
		   2: "matmul_compiled_naive",
		   3: "matmul_compiled_numpy"}


def curves(start, stop, step, funcs = funcs):
	
	sizes = np.arange(start, stop, step)
	times = np.zeros((4, len(sizes)))

	for i, func in enumerate(funcs) : 
		for j, size  in enumerate(sizes):
			A = ((np.random.random((size,size))-0.5)*0.1).astype("float32")
			B = ((np.random.random((size,size))-0.5)-0.1).astype("float32")
			C =np.zeros((size,size), dtype="float32")	
			t_start = time.time()		
			func(A, B, C, size, size, size)
			ellapsed_time = time.time() - t_start
			times[i][j] = ellapsed_time
	
	flops = sizes**3/times
	
	fig, ax = plt.subplots(figsize=(8, 5))
	for i in range(4):
		ax.plot(sizes, flops[i], label = funcs_d[i])
		ax.legend()
	ax.set_title('Numerical efficiency')
	ax.set_xlabel('Matrix size')
	ax.set_ylabel('GFLOPS')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='upper right')

	plt.show()
	return

curves(128, 1920, 128)


# (frugal_ai) ndreesen@Nicholass-MacBook-Air frugal_ai % uv run python -m matmul
# 0.64933180809021
# (frugal_ai) ndreesen@Nicholass-MacBook-Air frugal_ai % uv run python -m matmul
# 0.49695706367492676
# (frugal_ai) ndreesen@Nicholass-MacBook-Air frugal_ai % uv run python -m matmul
# 0.254716157913208
# (frugal_ai) ndreesen@Nicholass-MacBook-Air frugal_ai % uv run python -m matmul
# 0.25675487518310547