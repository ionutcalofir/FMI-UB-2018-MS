#include "matrix.h"

__global__ void convGPU_kernel(Matrix A, Matrix B, Matrix C, int startc) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int zz = startc + threadIdx.z;

	if (zz < A.channels) {

		int new_row = row - B.height + 2;
		int new_col = col - B.width + 2;

		if (new_row >= 0 && new_col >= 0 && new_row < C.height && new_col < C.width) {
			for (int i = row - 1; i <= row + 1; i++) {
				for (int j = col - 1; j <= col + 1; j++) {
					atomicAdd(&C.data[new_row * C.width + new_col], A.data[zz * A.height * A.width + i * A.width + j] * B.data[zz * B.height * B.width + (i - (row - 1)) * B.width + j - (col - 1)]);
				}
			}
		}
	}
}

void convGPU(Matrix input, Matrix kernel, Matrix output) {
	dim3 block(7, 7, 20); // creates a block of 7x7x20 threads
	dim3 grid(input.width / 7, input.height / 7, 1); // creats a grid of 4 x 4 blocks
	for (int i = 0; i < input.channels; i += 20) {
		convGPU_kernel<<<grid, block>>>(input, kernel, output, i);
	}
}
