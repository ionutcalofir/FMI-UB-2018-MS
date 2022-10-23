import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import engine
import utils.util as utils

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('--phase', default='search',
                      choices=['search', 'build_kmeans', 'build_db_bovw', 'build_kmeans_index', 'build_db_bovw_index', 'build_imgs_descriptors_index', 'compute_map'])
    args.add_argument('--query_path', default='../queries/45/45_11.jpg')
    args.add_argument('--db_path', default='../database')

    args.add_argument('--verbose', type=int, default=0)

    return args.parse_args()

def build_kmeans(db_path):
  engine.build_kmeans(db_path)

def build_db_bovw(db_path):
  engine.build_db_bovw(db_path)

def find_images(query_path, db_path):
  engine.find_images(query_path, db_path)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs('db_bovw', exist_ok=True)
    os.makedirs('faiss_index', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    os.makedirs('result_ransac', exist_ok=True)

    if args.phase == 'build_kmeans':
      build_kmeans(args.db_path)
    elif args.phase == 'build_db_bovw':
      build_db_bovw(args.db_path)
    elif args.phase == 'search':
      find_images(args.query_path, args.db_path)
    elif args.phase == 'build_kmeans_index':
      engine.build_kmeans_index()
    elif args.phase == 'build_db_bovw_index':
      engine.build_db_bovw_index()
    elif args.phase == 'build_imgs_descriptors_index':
      engine.build_imgs_descriptors_index(args.db_path)
    elif args.phase == 'compute_map':
      dirs = sorted(os.listdir('../queries'), key=lambda x: int(x))

      path = '../queries'
      maprec = 0
      for query in dirs:
        print('Query: {}'.format(query))
        path_query = path + '/' + query + '/' + query + '_11.jpg'
        retrieved_imgs = engine.find_images(path_query, args.db_path)

        retrieved_imgs = retrieved_imgs[:10]
        prec = 0
        for img in retrieved_imgs:
          cls = img.split('/')[2]

          if query == cls:
            prec += 1

        prec /= 10;
        maprec += prec

      maprec /= 50
      print('MAP: {}'.format(maprec))

      import pdb; pdb.set_trace()
