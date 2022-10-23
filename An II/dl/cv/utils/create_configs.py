import argparse
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def create_configs(args):
  label_id = {}
  with open(args.raw_config_path, 'r') as f:
    for i, line in enumerate(f):
      if i == 0:
        continue

      label = line.strip().split(',')[0]
      id = line.strip().split(',')[1]

      label_id[label] = id

  imgs = sorted(os.listdir(args.imgs_path))

  train_test_img = []
  train_test_label = []

  for img in imgs:
    if img[:-4] in label_id:
      train_test_img.append(img)
      train_test_label.append(int(label_id[img[:-4]]))
    else:
      with open(args.submission_test_txt_path, 'a') as f:
        f.write('{}\n'.format(img))

  train_img, test_img, train_label, test_label = train_test_split(train_test_img, train_test_label,
                                                                  test_size=0.3, random_state=42,
                                                                  shuffle=True, stratify=train_test_label)

  print('Weights: {}'.format(len(train_img) / (2 * np.bincount(train_label))))

  with open(args.test_txt_path, 'w') as f:
    for img, label in zip(test_img, test_label):
      f.write('{} {}\n'.format(img, label))

  with open(args.train_txt_path, 'w') as f:
    for img, label in zip(train_img, train_label):
      f.write('{} {}\n'.format(img, label))

  if os.path.isdir(args.cv_dir_path):
    shutil.rmtree(args.cv_dir_path)
  os.makedirs(args.cv_dir_path)

  train_img = np.array(train_img)
  train_label = np.array(train_label)
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
  for i, (train_idx, test_idx) in enumerate(skf.split(train_img, train_label)):
    X_train, X_test = train_img[train_idx], train_img[test_idx]
    y_train, y_test = train_label[train_idx], train_label[test_idx]

    fold_path = os.path.join(args.cv_dir_path, 'fold_{}'.format(i + 1))
    os.makedirs(fold_path)

    with open(fold_path + '/train.txt', 'w') as f:
      for img, label in zip(X_train, y_train):
        f.write('{} {}\n'.format(img, label))

    with open(fold_path + '/test.txt', 'w') as f:
      for img, label in zip(X_test, y_test):
        f.write('{} {}\n'.format(img, label))

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--raw_config_path', type=str,
      default='./data/configs/raw_configs/train_labels.txt')
  parser.add_argument('--imgs_path', type=str,
      default='./data/images')
  parser.add_argument('--submission_test_txt_path', type=str,
      default='./data/configs/submission_test/submission_test.txt')
  parser.add_argument('--train_txt_path', type=str,
      default='./data/configs/train/train.txt')
  parser.add_argument('--test_txt_path', type=str,
      default='./data/configs/test/test.txt')
  parser.add_argument('--cv_dir_path', type=str,
      default='./data/configs/cv/cv')

  return parser.parse_args()

def main(args):
  open(args.train_txt_path, 'w').close()
  open(args.submission_test_txt_path, 'w').close()

  create_configs(args)

if __name__ == '__main__':
  args = parse_args()
  main(args)
