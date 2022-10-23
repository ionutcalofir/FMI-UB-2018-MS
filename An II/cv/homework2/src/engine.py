from shutil import copyfile
import time
import os
import faiss
import numpy as np
import pickle
from sklearn.cluster import KMeans, MiniBatchKMeans

import utils.util as utils

def get_image_descriptors(query_path, verbose=0):
  query_original = utils.read_image(query_path)

  query = utils.erode(query_original, kernel_size=(5, 5))
  query = utils.blur(query, kernel_size=(7, 7))
  query_frames, query_descriptors = utils.get_sift_features(query)

  return query_descriptors

def get_bovw(img_descriptors, n_clusters=100000):
  t = time.time()

  # path = 'model/model_minibatch_kmeans_100000.pkl'
  # kmeans = pickle.load(open(path, 'rb'))
  # clusters_pred = kmeans.predict(img_descriptors)
  # bovw = np.zeros((clusters_pred.size, n_clusters))
  # bovw[np.arange(clusters_pred.size), clusters_pred] = 1
  # bovw = np.expand_dims(np.sum(bovw, axis=0), axis=0)
  # print('Done kmeans, time: {}'.format(time.time() - t))

  t = time.time()
  index = faiss.read_index('./faiss_index/kmeans_index.index')
  distances, best_labels = index.search(img_descriptors, 1)
  best_labels = best_labels[:, 0]
  bovw = np.zeros((best_labels.size, n_clusters))
  bovw[np.arange(best_labels.size), best_labels] = 1
  bovw = np.expand_dims(np.sum(bovw, axis=0), axis=0)
  print('Done faiss, time: {}'.format(time.time() - t))

  return bovw

def build_kmeans_index():
    path = 'model/model_minibatch_kmeans_100000.pkl'
    kmeans = pickle.load(open(path, 'rb'))

    embeddings = kmeans.cluster_centers_
    labels = np.array([i for i in range(kmeans.cluster_centers_.shape[0])], dtype=np.int64)
    index = utils.create_index(embeddings, labels)

    faiss.write_index(index, './faiss_index/kmeans_index.index')

def build_kmeans(db_path):
  dirs = sorted(os.listdir(db_path), key=lambda x: int(x))

  X = np.empty((0, 128), dtype=np.float32)
  for cls in dirs:
    print('Process class: {}/{}'.format(cls, len(dirs)))
    cls_path = os.path.join(db_path, cls)
    imgs = sorted(os.listdir(cls_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

    for img in imgs:
      img_path = os.path.join(cls_path, img)

      img_descriptors = get_image_descriptors(img_path)

      X = np.append(X, img_descriptors, axis=0)

  n_clusters = 10000
  minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=42,
                           verbose=10,
                           n_init=3,
                           init_size=3 * n_clusters,
                           init='random',
                           batch_size=10000)
  minibatch_kmeans.fit(X)
  model_name = 'model/model_minibatch_kmeans_10000_random.pkl'
  pickle.dump(minibatch_kmeans, open(model_name, 'wb'))

  n_clusters = 100000
  minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=42,
                           verbose=10,
                           n_init=3,
                           init_size=3 * n_clusters,
                           init='random',
                           batch_size=10000)
  minibatch_kmeans.fit(X)
  model_name = 'model/model_minibatch_kmeans_100000_random.pkl'
  pickle.dump(minibatch_kmeans, open(model_name, 'wb'))

  n_clusters = 10000
  minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=42,
                           verbose=10,
                           n_init=3,
                           init_size=3 * n_clusters,
                           init='k-means++',
                           batch_size=10000)
  minibatch_kmeans.fit(X)
  model_name = 'model/model_minibatch_kmeans_10000.pkl'
  pickle.dump(minibatch_kmeans, open(model_name, 'wb'))

  n_clusters = 100000
  minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           random_state=42,
                           verbose=10,
                           n_init=3,
                           init_size=3 * n_clusters,
                           init='k-means++',
                           batch_size=10000)
  minibatch_kmeans.fit(X)
  model_name = 'model/model_minibatch_kmeans_100000.pkl'
  pickle.dump(minibatch_kmeans, open(model_name, 'wb'))

def get_tf(X):
  return X / np.expand_dims(np.sum(X, axis=1), axis=-1)

def get_idf(X):
  return np.log(500 / (1 + np.expand_dims(np.sum(X > 0, axis=0), axis=0)))

def get_tf_idf(X):
  tf = get_tf(X)
  idf = get_idf(X)

  tf_idf = tf * idf
  return tf_idf

def normalize(X):
  norm = np.expand_dims(np.linalg.norm(X, axis=1), axis=-1)
  return X / norm

def build_db_bovw(db_path):
  dirs = sorted(os.listdir(db_path), key=lambda x: int(x))
  imgs_path = []

  X = np.empty((0, 100000))
  for cls in dirs:
    print('Process class: {}/{}'.format(cls, len(dirs)))
    cls_path = os.path.join(db_path, cls)
    imgs = sorted(os.listdir(cls_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

    for img in imgs:
      img_path = os.path.join(cls_path, img)
      imgs_path.append(img_path)

      img_descriptors = get_image_descriptors(img_path)
      bovw = get_bovw(img_descriptors)
      X = np.append(X, bovw, axis=0)

  idf = get_idf(X)
  tf_idf = get_tf_idf(X)
  tf_idf = normalize(tf_idf)

  path = 'db_bovw/db_bovw_tfidf.pkl'
  pickle.dump(tf_idf, open(path, 'wb'))
  path = 'db_bovw/db_bovw_idf.pkl'
  pickle.dump(idf, open(path, 'wb'))

def build_db_bovw_index():
  path = 'db_bovw/db_bovw_tfidf.pkl'
  db_bovw_tfidf = pickle.load(open(path, 'rb')).astype(np.float32)
  labels = np.array([i for i in range(db_bovw_tfidf.shape[0])], dtype=np.int64)

  index = utils.create_index(db_bovw_tfidf, labels)
  faiss.write_index(index, './faiss_index/db_bovw_index.index')

def build_imgs_descriptors_index(db_path):
  dirs = sorted(os.listdir(db_path), key=lambda x: int(x))

  for cls in dirs:
    print('Process class: {}/{}'.format(cls, len(dirs)))
    cls_path = os.path.join(db_path, cls)
    imgs = sorted(os.listdir(cls_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

    for img in imgs:
      img_path = os.path.join(cls_path, img)

      img_descriptors = get_image_descriptors(img_path)
      labels = np.array([i for i in range(img_descriptors.shape[0])], dtype=np.int64)
      index = utils.create_index(img_descriptors, labels)
      img_name = img_path.split('/')[-1][:-4]
      faiss.write_index(index, './faiss_index/imgs/{}.index'.format(img_name))

def get_inliers(query_path, template_path, img_name, verbose=0):
    query_original = utils.read_image(query_path)
    template_original = utils.read_image(template_path)

    query = utils.erode(query_original, kernel_size=(5, 5))
    template = utils.erode(template_original, kernel_size=(5, 5))

    query = utils.blur(query, kernel_size=(7, 7))
    template = utils.blur(template, kernel_size=(7, 7))

    query_frames, query_descriptors = utils.get_sift_features(query)
    template_frames, template_descriptors = utils.get_sift_features(template)

    index = faiss.read_index('./faiss_index/imgs/{}.index'.format(img_name))
    distances, best_labels = index.search(template_descriptors, 2)
    good_matches = utils.preprocess_matches(distances, best_labels)

    query_pts = np.array([[query_frames[m[1]][1], query_frames[m[1]][0]] for m in good_matches]) # (X, Y)
    template_pts = np.array([[template_frames[m[0]][1], template_frames[m[0]][0]] for m in good_matches])

    # flann = utils.get_flann_kdtree()
    # matches = flann.knnMatch(query_descriptors, template_descriptors, k=2)
    # good_matches = utils.preprocess_matches(matches)

    # query_pts = np.array([[query_frames[m.queryIdx][1], query_frames[m.queryIdx][0]] for m in good_matches]) # (X, Y)
    # template_pts = np.array([[template_frames[m.trainIdx][1], template_frames[m.trainIdx][0]] for m in good_matches])

    M, mask = utils.get_homography(query_pts, template_pts)
    inliers = np.sum(mask)

    return inliers

def find_images(query_path, db_path):
  t = time.time()
  index = faiss.read_index('./faiss_index/db_bovw_index.index')

  path = 'db_bovw/db_bovw_idf.pkl'
  idf = pickle.load(open(path, 'rb')).astype(np.float32)

  img_descriptors = get_image_descriptors(query_path)
  img_bovw = get_bovw(img_descriptors)
  img_bovw_tfidf = get_tf(img_bovw) * idf
  img_bovw_tfidf = normalize(img_bovw_tfidf).astype(np.float32)

  neighbours = 25
  distances, best_labels = index.search(img_bovw_tfidf, neighbours)
  matches = best_labels[0, :]

  dirs = sorted(os.listdir(db_path), key=lambda x: int(x))
  imgs_path = []
  for cls in dirs:
    cls_path = os.path.join(db_path, cls)
    imgs = sorted(os.listdir(cls_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

    for img in imgs:
      img_path = os.path.join(cls_path, img)
      imgs_path.append(img_path)
  imgs_path = np.array(imgs_path)

  print('Retrieved images:')
  retrieved_imgs = imgs_path[matches]
  for img_path in retrieved_imgs:
    print(img_path)

  print('Total time: {}'.format(time.time() - t))
  copyfile(query_path, './result/query.jpg')
  for i, img_path in enumerate(retrieved_imgs):
    copyfile(img_path, './result/match_{}.jpg'.format(i))

  print('Begin RANSAC')
  t = time.time()
  ransac_inliers = []
  retrieved_imgs = imgs_path[matches]
  for i, img_path in enumerate(retrieved_imgs):
    print('Process img: {}'.format(i))
    img_name = img_path.split('/')[-1][:-4]
    inliers = get_inliers(img_path, query_path, img_name)
    ransac_inliers.append(inliers)
  sort_ransac = np.argsort(ransac_inliers)[::-1]
  retrieved_imgs = retrieved_imgs[sort_ransac]
  for img_path in retrieved_imgs:
    print(img_path)
  print('RANSAC time: {}'.format(time.time() - t))
  copyfile(query_path, './result_ransac/query.jpg')
  for i, img_path in enumerate(retrieved_imgs):
    copyfile(img_path, './result_ransac/match_{}.jpg'.format(i))

  return retrieved_imgs
