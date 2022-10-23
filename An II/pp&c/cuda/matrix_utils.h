#pragma once

#include <fstream>
#include "matrix.h"

void init(Matrix& M, int height, int width, int channels) {
  M.height = height;
  M.width = width;
  M.channels = channels;
}

void saveData(Matrix M, std::string name) {
  std::ofstream fout_img_height(name + "_height.bin", std::ios::out | std::ios::binary);
  fout_img_height.write(reinterpret_cast<char*>(&M.height), sizeof(M.height));
  fout_img_height.close();

  std::ofstream fout_img_width(name + "_width.bin", std::ios::out | std::ios::binary);
  fout_img_width.write(reinterpret_cast<char*>(&M.width), sizeof(M.width));
  fout_img_width.close();

  std::ofstream fout_img_channels(name + "_channels.bin", std::ios::out | std::ios::binary);
  fout_img_channels.write(reinterpret_cast<char*>(&M.channels), sizeof(M.channels));
  fout_img_channels.close();

  std::ofstream fout_img(name + ".bin", std::ios::out | std::ios::binary);
  fout_img.write(reinterpret_cast<char*>(M.data), sizeof(int) * (M.height * M.width * M.channels));
  fout_img.close();
}

Matrix loadData(std::string name) {
  Matrix M;

  std::ifstream fin_img_height(name + "_height.bin", std::ios::in | std::ios::binary);
  fin_img_height.read(reinterpret_cast<char*>(&M.height), sizeof(M.height));
  fin_img_height.close();

  std::ifstream fin_img_width(name + "_width.bin", std::ios::in | std::ios::binary);
  fin_img_width.read(reinterpret_cast<char*>(&M.width), sizeof(M.width));
  fin_img_width.close();

  std::ifstream fin_img_channels(name + "_channels.bin", std::ios::in | std::ios::binary);
  fin_img_channels.read(reinterpret_cast<char*>(&M.channels), sizeof(M.channels));
  fin_img_channels.close();

  M.data = new int[M.height * M.width * M.channels];

  std::ifstream fin_img(name + ".bin", std::ios::in | std::ios::binary);
  fin_img.read(reinterpret_cast<char*>(M.data), sizeof(int) * (M.height * M.width * M.channels));
  fin_img.close();

  return M;
}
