#pragma once

#include <fstream>

struct Matrix {
  int height, width, channels;
  int *data;
};

extern "C" void convGPU(Matrix input, Matrix kernel, Matrix output);
extern "C" void maxPoolGPU(Matrix input, Matrix output);
extern "C" void fcGPU(Matrix input, int* weights, Matrix output);
