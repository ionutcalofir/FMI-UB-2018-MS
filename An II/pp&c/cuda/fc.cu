#include "matrix.h"

__global__ void fcGPU_kernel(Matrix input, int* weights, Matrix output) {
	int idx = threadIdx.x;

	/*
	 *output.data[0] = input.data[idx];
	 */
	/*
	 *output.data[0] = weights[idx];
	 */
	atomicAdd(&output.data[0], input.data[idx] * weights[idx]);
}

void fcGPU(Matrix input, int* weights, Matrix output) {
	dim3 block(169, 1, 1); // creates a block of 169 threads
	dim3 grid(1, 1, 1); // creats a grid of 1 blocks
	fcGPU_kernel<<<grid, block>>>(input, weights, output);
}
