import argparse
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def create_configs(args):
  X = []
  with open(args.raw_train_config_path, 'r') as f:
    for line in f:
      X.append(line.strip())

  X = np.array(X)
  X_train, X_test = train_test_split(X, shuffle=True, random_state=42)

  with open(args.train_config_path, 'w') as f:
    for text in X_train:
      f.write('{}\n'.format(text))
  with open(args.test_config_path, 'w') as f:
    for text in X_test:
      f.write('{}\n'.format(text))

  if os.path.isdir(args.cv_dir_path):
    shutil.rmtree(args.cv_dir_path)
  os.makedirs(args.cv_dir_path)

  kf = KFold(n_splits=3, shuffle=True, random_state=42)
  for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    train_text = X[train_idx]
    test_text = X[test_idx]

    fold_path = os.path.join(args.cv_dir_path, 'fold_{}'.format(i + 1))
    os.makedirs(fold_path)

    with open(fold_path + '/train.txt', 'w') as f:
      for text in train_text:
        f.write('{}\n'.format(text))
    with open(fold_path + '/test.txt', 'w') as f:
      for text in test_text:
        f.write('{}\n'.format(text))

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--raw_train_config_path',
      default='./data/configs/raw_configs/train_full.txt')
  parser.add_argument('--train_config_path',
      default='./data/configs/train/train.txt')
  parser.add_argument('--test_config_path',
      default='./data/configs/test/test.txt')
  parser.add_argument('--cv_dir_path',
      default='./data/configs/cv/cv')

  return parser.parse_args()

def main(args):
  create_configs(args)

if __name__ == '__main__':
  args = parse_args()
  main(args)
