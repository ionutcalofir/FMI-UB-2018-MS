import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_TH = 1000000

def main():
  imgs = os.listdir('./data/images')
  imgs = [os.path.join('./data', 'images', img) for img in imgs]

  if os.path.isdir('./data/images_processed_good_{}'.format(IMG_TH)):
    shutil.rmtree('./data/images_processed_good_{}'.format(IMG_TH))
  if os.path.isdir('./data/images_processed_bad_{}'.format(IMG_TH)):
    shutil.rmtree('./data/images_processed_bad_{}'.format(IMG_TH))
  os.makedirs('./data/images_processed_good_{}'.format(IMG_TH))
  os.makedirs('./data/images_processed_bad_{}'.format(IMG_TH))

  for i, img_path in enumerate(imgs):
    print('Preprocess img: {}/{}'.format(i, len(imgs)))
    img = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]

    if np.sum(img) > IMG_TH:
      shutil.copyfile(os.path.join('./data', 'images', img_name),
                      os.path.join('./data', 'images_processed_good_{}'.format(IMG_TH), img_name))
    else:
      shutil.copyfile(os.path.join('./data', 'images', img_name),
                      os.path.join('./data', 'images_processed_bad_{}'.format(IMG_TH), img_name))

if __name__ == '__main__':
  main()
