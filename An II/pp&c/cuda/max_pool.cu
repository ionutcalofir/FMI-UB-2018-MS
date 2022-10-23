#include "matrix.h"
#include <algorithm>

__global__ void maxPoolGPU_kernel(Matrix input, Matrix output) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	extern __shared__ int s[];

	s[threadIdx.y * 2 + threadIdx.x] = input.data[row * input.width + col];

	__syncthreads();

	int new_row = row / 2;
	int new_col = col / 2;
	atomicMax(&output.data[new_row * output.width + new_col], s[threadIdx.y * 2 + threadIdx.x]);
	/*
	 *output.data[new_row * output.width + new_col] = max(s[0], max(s[1], max(s[2], s[3])));
	 */
}

void maxPoolGPU(Matrix input, Matrix output) {
	dim3 block(2, 2, 1); // creates a block of 7x7x1 threads
	dim3 grid(input.width / 2, input.height / 2, 1); // creats a grid of 4 x 4 blocks
	maxPoolGPU_kernel<<<grid, block, sizeof(int) * 2 * 2>>>(input, output);
}
