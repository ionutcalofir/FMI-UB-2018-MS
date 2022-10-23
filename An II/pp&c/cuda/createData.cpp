#include "matrix_utils.h"

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

using namespace std;

const int MAX_HEIGHT = 28, MAX_WIDTH = 28, MAX_CHANNELS = 512;
const int K_MAX_HEIGHT = 3, K_MAX_WIDTH = 3, K_MAX_CHANNELS = 512;

int main() {
  /*
   *srand (time(nullptr));
   */

  Matrix img, kernel;
  img.height = MAX_HEIGHT;
  img.width = MAX_WIDTH;
  img.channels = MAX_CHANNELS;
  img.data = new int[img.height * img.width * img.channels];
  for (int i = 0; i < MAX_HEIGHT; i++) {
    for (int j = 0; j < MAX_WIDTH; j++) {
      for (int z = 0; z < MAX_CHANNELS; z++) {
        int nr = rand() % 10 + 1;
        img.data[z * img.width * img.height + i * img.width + j] = nr;
      }
    }
  }
  kernel.height = K_MAX_HEIGHT;
  kernel.width = K_MAX_WIDTH;
  kernel.channels = K_MAX_CHANNELS;
  kernel.data = new int[kernel.height * kernel.width * kernel.channels];
  /*
   *cout << kernel.height * kernel.width * kernel.channels << '\n';
   *cout << sizeof(kernel.data) << ' ' << sizeof(kernel.height) << ' ' << sizeof(kernel.channels) << ' ' << sizeof(kernel.width) << '\n';
   *cout << sizeof(kernel) << '\n';
   */
  for (int i = 0; i < K_MAX_HEIGHT; i++) {
    for (int j = 0; j < K_MAX_WIDTH; j++) {
      for (int z = 0; z < K_MAX_CHANNELS; z++) {
        int nr = rand() % 10 + 1;
        kernel.data[z * kernel.width * kernel.height + i * kernel.width + j] = nr;
      }
    }
  }

  saveData(img, "image");
  Matrix read_img = loadData("image");

  saveData(kernel, "kernel");
  Matrix read_kernel = loadData("kernel");

  int sum_img = 0;
  for (int i = 0; i < MAX_HEIGHT; i++) {
    for (int j = 0; j < MAX_WIDTH; j++) {
      for (int z = 0; z < MAX_CHANNELS; z++) {
        sum_img += img.data[z * img.width * img.height + i * img.width + j] - read_img.data[z * img.width * img.height + i * img.width + j];
      }
    }
  }
  assert(sum_img == 0);

  int sum_kernel = 0;
  for (int i = 0; i < K_MAX_HEIGHT; i++) {
    for (int j = 0; j < K_MAX_WIDTH; j++) {
      for (int z = 0; z < K_MAX_CHANNELS; z++) {
        sum_kernel += kernel.data[z * kernel.width * kernel.height + i * kernel.width + j] - read_kernel.data[z * kernel.width * kernel.height + i * kernel.width + j];
      }
    }
  }
  assert(sum_kernel == 0);

  return 0;
}
