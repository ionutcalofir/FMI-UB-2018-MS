#include "cuda_runtime.h"
#include "matrix_utils.h"

#include <fstream>
#include <iostream>
#include <string.h>
#include <chrono> 
#include <bits/stdc++.h>

using namespace std;

const int MAX_HEIGHT = 28, MAX_WIDTH = 28, MAX_CHANNELS = 512;
const int K_MAX_HEIGHT = 3, K_MAX_WIDTH = 3, K_MAX_CHANNELS = 512;

void conv_cpu(Matrix img, Matrix kernel, Matrix output_cpu) {
  for (int row = 0; row < img.height; row++) {
    int new_row = row - kernel.width + 2;
    if (new_row < 0 || new_row >= 26) {
      continue;
    }
    for (int col = 0; col < img.width; col++) {
      int new_col = col - kernel.height + 2;
      if (new_col < 0 || new_col >= 26) {
        continue;
      }

      for (int i = row - 1; i <= row + 1; i++) {
        for (int j = col - 1; j <= col + 1; j++) {
          for (int z = 0; z < img.channels; z++) {
            output_cpu.data[new_row * output_cpu.width + new_col] += img.data[z * img.height * img.width + i * img.width + j] * kernel.data[z * kernel.height * kernel.width + (i - (row - 1)) * kernel.width + j - (col - 1)];
          }
        }
      }
    }
  }
}

void max_pool_cpu(Matrix input, Matrix output) {
  for (int i = 0; i < input.height; i++) {
    for (int j = 0; j < input.width; j++) {
      int new_i = i / 2;
      int new_j = j / 2;
      output.data[new_i * output.width + new_j] = max(output.data[new_i * output.width + new_j],
                                                      input.data[i * input.width + j]);
    }
  }
}

void fc_cpu(Matrix input, const int* weights, Matrix output) {
  for (int i = 0; i < input.width * input.height; i++) {
    output.data[0] += input.data[i] * weights[i];
  }
}

int main() {
  Matrix img, kernel;
  img = loadData("image");
  kernel = loadData("kernel");

  Matrix g_img, g_kernel;
  init(g_img, img.height, img.width, img.channels);
  cudaMalloc(&g_img.data, sizeof(int) * g_img.height * g_img.width * g_img.channels);
  cudaMemcpy(g_img.data, img.data, sizeof(int) * (img.height * img.width * img.channels), cudaMemcpyHostToDevice);
  init(g_kernel, kernel.height, kernel.width, kernel.channels);
  cudaMalloc(&g_kernel.data, sizeof(int) * g_kernel.height * g_kernel.width * g_kernel.channels);
  cudaMemcpy(g_kernel.data, kernel.data, sizeof(int) * (kernel.height * kernel.width * kernel.channels), cudaMemcpyHostToDevice);

  // CONV

  Matrix g_output;
  init(g_output, 26, 26, 1);
  cudaMalloc(&g_output.data, sizeof(int) * 26 * 26 * 1);
  cudaMemset(&g_output.data, 0, sizeof(int) * 26 * 26 * 1);

  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);

  cudaEventRecord(startGPU);
  convGPU(g_img, g_kernel, g_output);
  cudaEventRecord(stopGPU);

  Matrix output;
  init(output, 26, 26, 1);
  output.data = (int*) malloc(sizeof(int) * 26 * 26 * 1);
  cudaMemcpy(output.data, g_output.data, sizeof(int) * 26 * 26 * 1, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stopGPU);

  float miliseconds = 0.0;
  cudaEventElapsedTime(&miliseconds, startGPU, stopGPU);
  cout << "Time elapsed for conv on GPU: " << miliseconds * 1e-3 << " seconds." << '\n';
  /*
   *for (int i = 0; i < 26; i++) {
   *  for (int j = 0; j < 26; j++) {
   *    cout << output.data[i * 26 + j] << ' ';
   *  }
   *  cout << '\n';
   *}
   */

  Matrix output_cpu;
  init(output_cpu, 26, 26, 1);
  output_cpu.data = (int*) malloc(sizeof(int) * 26 * 26 * 1);
  memset(output_cpu.data, 0, sizeof(int) * 26 * 26 * 1);

  auto startCPU = chrono::high_resolution_clock::now(); 
  conv_cpu(img, kernel, output_cpu);
  auto endCPU = chrono::high_resolution_clock::now(); 
  double duration = chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU).count(); 
  cout << "Time elapsed for conv on CPU: " << duration * 1e-3 << " seconds." << '\n';
  /*
   * cout << '\n';
   *for (int i = 0; i < 26; i++) {
   *  for (int j = 0; j < 26; j++) {
   *    cout << output_cpu.data[i * 26 + j] << ' ';
   *  }
   *  cout << '\n';
   *}
   */

  // MAX POOL

  Matrix g_output_maxpool;
  init(g_output_maxpool, 13, 13, 1);
  cudaMalloc(&g_output_maxpool.data, sizeof(int) * 13 * 13 * 1);
  cudaMemset(&g_output_maxpool.data, 0, sizeof(int) * 13 * 13 * 1);

  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);

  cudaEventRecord(startGPU);
  maxPoolGPU(g_output, g_output_maxpool);
  cudaEventRecord(stopGPU);

  Matrix output_maxpool;
  init(output_maxpool, 13, 13, 1);
  output_maxpool.data = (int*) malloc(sizeof(int) * 13 * 13 * 1);
  cudaMemcpy(output_maxpool.data, g_output_maxpool.data, sizeof(int) * 13 * 13 * 1, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stopGPU);

  miliseconds = 0.0;
  cudaEventElapsedTime(&miliseconds, startGPU, stopGPU);
  cout << "Time elapsed for max pool on GPU: " << miliseconds * 1e-3 << " seconds." << '\n';
  /*
   *for (int i = 0; i < 13; i++) {
   *  for (int j = 0; j < 13; j++) {
   *    cout << output_maxpool.data[i * 13 + j] << ' ';
   *  }
   *  cout << '\n';
   *}
   */

  Matrix output_maxpool_cpu;
  init(output_maxpool_cpu, 13, 13, 1);
  output_maxpool_cpu.data = (int*) malloc(sizeof(int) * 13 * 13 * 1);
  memset(output_maxpool_cpu.data, 0, sizeof(int) * 13 * 13 * 1);

  startCPU = chrono::high_resolution_clock::now(); 
  max_pool_cpu(output_cpu, output_maxpool_cpu);
  endCPU = chrono::high_resolution_clock::now(); 
  duration = chrono::duration_cast<chrono::nanoseconds>(endCPU - startCPU).count(); 
  cout << "Time elapsed for max pool on CPU: " << duration * 1e-9 << " seconds." << '\n';

  /*
   *cout << '\n';
   *for (int i = 0; i < 13; i++) {
   *  for (int j = 0; j < 13; j++) {
   *    cout << output_maxpool_cpu.data[i * 13 + j] << ' ';
   *  }
   *  cout << '\n';
   *}
   */

  // FC
  int *weights;
  weights = (int*) malloc(sizeof(int) * 169 * 1 * 1);
  for (int i = 0; i < 169; i++) {
    weights[i] = i % 5;
  }

  int* weights_gpu;
  cudaMalloc(&weights_gpu, sizeof(int) * 169 * 1 * 1);
  cudaMemcpy(weights_gpu, weights, sizeof(int) * 169 * 1 * 1, cudaMemcpyHostToDevice);

  Matrix g_output_fc;
  init(g_output_fc, 1, 1, 1);
  cudaMalloc(&g_output_fc.data, sizeof(int) * 1 * 1 * 1);
  cudaMemset(&g_output_fc.data, 0, sizeof(int) * 1 * 1 * 1);

  cudaEventCreate(&startGPU);
  cudaEventCreate(&stopGPU);

  cudaEventRecord(startGPU);
  fcGPU(g_output_maxpool, weights_gpu, g_output_fc);
  cudaEventRecord(stopGPU);

  Matrix output_fc;
  init(output_fc, 1, 1, 1);
  output_fc.data = (int*) malloc(sizeof(int) * 1 * 1 * 1);
  cudaMemcpy(output_fc.data, g_output_fc.data, sizeof(int) * 1 * 1 * 1, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stopGPU);

  miliseconds = 0.0;
  cudaEventElapsedTime(&miliseconds, startGPU, stopGPU);
  cout << "Time elapsed for max pool on GPU: " << miliseconds * 1e-3 << " seconds." << '\n';

  /*
   *cout << output_fc.data[0] << '\n';
   */

  Matrix output_fc_cpu;
  init(output_fc_cpu, 1, 1, 1);
  output_fc_cpu.data = (int*) malloc(sizeof(int) * 1 * 1 * 1);
  memset(output_fc_cpu.data, 0, sizeof(int) * 1 * 1 * 1);

  startCPU = chrono::high_resolution_clock::now(); 
  fc_cpu(output_maxpool_cpu, weights, output_fc_cpu);
  endCPU = chrono::high_resolution_clock::now(); 
  duration = chrono::duration_cast<chrono::nanoseconds>(endCPU - startCPU).count(); 
  cout << "Time elapsed for max pool on CPU: " << duration * 1e-9 << " seconds." << '\n';

  /*
   *cout << output_fc_cpu.data[0] << '\n';
   */

  return 0;
}
