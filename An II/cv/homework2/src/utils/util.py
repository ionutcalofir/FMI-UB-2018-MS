import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cyvlfeat.sift import sift

def get_sift_features(image):
    """
    Apply vlfeat implementation of SIFT
    Github: https://github.com/menpo/cyvlfeat
    """
    frames, descriptors = sift(image, compute_descriptor=True, float_descriptors=True, verbose=False)
    return frames, descriptors

def read_image(image_path, rgb=False):
    """
    Read an image and convert it to grayscale
    """
    if rgb:
      image = Image.open(image_path)
    else:
      image = Image.open(image_path).convert('L')
    image = np.array(image, dtype=np.float32)
    return image


def erode(image, kernel_size=(5, 5)):
    """
    Apply erode to an image
    """
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.erode(image, kernel)
    return image


def blur(image, kernel_size=(7, 7)):
    """
    Apply blur to an image
    """
    image = cv2.blur(image, kernel_size)
    return image


def create_index(embeddings, labels):
    num_elements, dim = embeddings.shape
    nlist = len(labels)
    quantizer = faiss.IndexFlatL2(dim)

    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.nprobe = 25
    assert not index.is_trained
    index.train(embeddings)
    assert index.is_trained

    index.add_with_ids(embeddings, labels)

    return index


def preprocess_matches(dist_matches, best_labels):
    good_matches = []
    for i, dist in enumerate(dist_matches):
        if np.sqrt(dist[0]) < 0.7 * np.sqrt(dist[1]):
            good_matches.append((i, best_labels[i][0])) # (template_pt, query_pt)

    return good_matches


# def preprocess_matches(matches):
    # """
    # Store all the good matches as per Lowe's ratio test.
    # Idea from: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    # """
    # good_matches = []
    # for m, n in matches:
        # if m.distance < 0.7 * n.distance:
            # good_matches.append(m)

    # return good_matches


def get_flann_kdtree():
    """
    Fast Library for Approximate Nearest Neighbors
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=16)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann


def get_homography(query_pts, template_pts):
    M, mask = cv2.findHomography(query_pts, template_pts, cv2.RANSAC, 5.0)
    return M, mask
