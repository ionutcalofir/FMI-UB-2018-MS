import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import engine
import utils.util as utils

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('--template_path', default='../data/image_1.jpg')
    args.add_argument('--query_path', default='../data/image_1.jpg')

    args.add_argument('--verbose', type=int, default=0)

    return args.parse_args()

def solve(query_path, template_path, verbose=0):
    image, image_rgb = engine.get_image_info(query_path, template_path, verbose)
    print('Done SIFT!')

    table1, table2, box_coords, grade_coords = engine.get_image_data(image, image_rgb, verbose=verbose)
    box_crop = utils.get_crop(image_rgb, box_coords[0], box_coords[1], box_coords[2], box_coords[3], rgb=True)
    grade_crop = utils.get_crop(image_rgb, grade_coords[0], grade_coords[1], grade_coords[2], grade_coords[3], rgb=True)

    if verbose:
      fig = plt.figure('Box Crop')
      plt.imshow(box_crop.astype(np.uint8))
      plt.savefig('logs/' + 'Box Crop' + '.png', bbox_inches='tight')
      plt.close(fig)

      fig = plt.figure('Grade Crop')
      plt.imshow(grade_crop.astype(np.uint8))
      plt.savefig('logs/' + 'Grade Crop' + '.png', bbox_inches='tight')
      plt.close(fig)

    box_number = utils.get_box_number(box_crop)
    grade_number = utils.get_grade_number(grade_crop)

    print('Table 1:')
    for k, v in table1.items():
        print('{}: {}'.format(k, v))
    print('Table 2:')
    for k, v in table2.items():
        print('{}: {}'.format(k, v))
    print('Box number: {}\nGrade: {}'.format(box_number, grade_number))

if __name__ == '__main__':
    args = parse_args()
    os.makedirs('logs', exist_ok=True)

    solve(args.query_path, args.template_path, args.verbose)
