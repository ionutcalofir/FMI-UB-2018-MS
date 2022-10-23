import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cyvlfeat.sift import sift

from mnist.train import predict_number_box, predict_number_grade

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


def dilate(image, iterations, kernel_size=(5, 5)):
    """
    Apply erode to an image
    """
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.dilate(image, kernel, iterations=iterations)
    return image


def blur(image, kernel_size=(7, 7)):
    """
    Apply blur to an image
    """
    image = cv2.blur(image, kernel_size)
    return image


def get_flann_kdtree():
    """
    Fast Library for Approximate Nearest Neighbors
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=16)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann


def preprocess_matches(matches):
    """
    Store all the good matches as per Lowe's ratio test.
    Idea from: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    """
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches


def get_homography(query_pts, template_pts):
    M, mask = cv2.findHomography(query_pts, template_pts, cv2.RANSAC, 5.0)
    return M, mask


def get_query_scale(homography_query_pts, homography_template_pts):
    homography_query_pts_repeat = np.repeat(homography_query_pts, homography_query_pts.shape[0], axis=0)
    homography_query_pts_tile = np.tile(homography_query_pts, (homography_query_pts.shape[0], 1))
    homography_query_pts_norm = np.linalg.norm(homography_query_pts_repeat - homography_query_pts_tile, axis=1)

    homography_template_pts_repeat = np.repeat(homography_template_pts, homography_template_pts.shape[0], axis=0)
    homography_template_pts_tile = np.tile(homography_template_pts, (homography_template_pts.shape[0], 1))
    homography_template_pts_norm = np.linalg.norm(homography_template_pts_repeat - homography_template_pts_tile, axis=1)

    homography_query_pts_norm = homography_query_pts_norm[homography_template_pts_norm != 0.]
    homography_template_pts_norm = homography_template_pts_norm[homography_template_pts_norm != 0.]

    scale = homography_query_pts_norm / homography_template_pts_norm
    scale = scale.sum() / scale.shape[0]

    return scale


def apply_perspective_transform(pts, M):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)


def warp_perspective(image, M, width, height):
    return cv2.warpPerspective(image, M, (width, height))


def sobel_x():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])


def sobel_y():
    return np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])


def apply_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def get_box_filter(kernel_size=(100, 100)):
    box_filter = np.zeros(kernel_size, dtype=np.float32)

    h, w = kernel_size
    for i in range(h):
        box_filter[i, 0] = 1.
        box_filter[i, w - 1] = 1.
    for j in range(w):
        box_filter[0, j] = 1.
        box_filter[h - 1, j] = 1.

    return box_filter


def apply_nms(image, verbose=False, kernel_size=(90, 90), fig_name='nms'):
    if verbose:
      bold_size = 10
      plot_image = np.zeros(image.shape, dtype=np.float32)

    new_image = np.zeros(image.shape, dtype=np.float32)

    max_inds = np.flip(np.argsort(image, axis=None))
    for pos in range(100000):
      max_ind = np.unravel_index(max_inds[pos], image.shape)
      if image[max_ind] == 0.:
        continue

      image[(max_ind[0] - kernel_size[0]):(max_ind[0] + kernel_size[0]),
            (max_ind[1] - kernel_size[1]):(max_ind[1] + kernel_size[1])] = 0.

      new_image[max_ind[0], max_ind[1]] = 255.

      if verbose:
        plot_image[(max_ind[0] - bold_size):(max_ind[0] + bold_size),
                   (max_ind[1] - bold_size):(max_ind[1] + bold_size)] = 255.

    if verbose:
      fig = plt.figure(fig_name)
      plt.imshow(plot_image, cmap='gray')
      # plt.show()
      plt.savefig('logs/' + fig_name + '.png', bbox_inches='tight')
      plt.close(fig)

    return new_image


def get_header_pos(image):
    y_th = 2500
    x_th = 2000
    inds_nonzero = np.nonzero(image)
    inds = [(inds_nonzero[0][i], inds_nonzero[1][i]) for i in range(inds_nonzero[0].shape[0])]

    inds = [(ind[0], ind[1]) for ind in inds if ind[0] > y_th]
    col1 = sorted([(ind[0], ind[1]) for ind in inds if ind[1] < x_th], key=lambda x: x[0])
    col2 = sorted([(ind[0], ind[1]) for ind in inds if ind[1] > x_th], key=lambda x: x[0])

    return col1[0], col2[0]


def get_xs(image, box_image, header_pos):
  lt_offset = 50
  rt_offset = 600
  up_offset = 100
  down_offset = 2000
  sum_th = 200000

  kernel_size = 30

  table = box_image[(header_pos[0] + up_offset):(header_pos[0] + down_offset),
                    (header_pos[1] - lt_offset):(header_pos[1] + rt_offset)]
  image_table = image[(header_pos[0] + up_offset):(header_pos[0] + down_offset),
                      (header_pos[1] - lt_offset):(header_pos[1] + rt_offset)]

  inds_nonzero = np.nonzero(table)
  inds = [(inds_nonzero[0][i], inds_nonzero[1][i]) for i in range(inds_nonzero[0].shape[0])]

  inds = sorted(inds, key=lambda x: x[0])
  tb = []
  for i in range(0, 60, 4):
    tb.append(sorted(inds[i:i + 4], key=lambda x: x[1]))

  out = {i: [] for i in range(1, 16)}

  for ex, row in enumerate(tb):
    for col, (h, w) in enumerate(row):
      if int(np.sum(image_table[(h - kernel_size):(h + kernel_size), (w - kernel_size):(w + kernel_size)])) >= sum_th:
        out[ex + 1].append(col + 1)

  return out


def get_crop(image, up, down, lt, rt, rgb=False):
  if rgb:
    crop = image[up:down, lt:rt, :]
  else:
    crop = image[up:down, lt:rt]

  return crop


def get_box_coords(image, box_image, coords_kernel_size=(75, 75)):
  lt_pos = 3450
  rt_pos = 3750
  up_pos = 2500
  down_pos = 3030
  kernel_size = 20

  crop_image = image[up_pos:down_pos, lt_pos:rt_pos]
  crop_box_image = box_image[up_pos:down_pos, lt_pos:rt_pos]

  inds_nonzero = np.nonzero(crop_box_image)
  inds = [(inds_nonzero[0][i], inds_nonzero[1][i]) for i in range(inds_nonzero[0].shape[0])]
  inds = sorted(inds, key=lambda x: x[0])

  mean_x = (inds[0][1] + inds[1][1]) // 2
  inds[0] = (inds[0][0], mean_x)
  inds[1] = (inds[1][0], mean_x)

  box1 = np.sum(crop_image[(inds[0][0] - kernel_size):(inds[0][0] + kernel_size), (inds[0][1] - kernel_size):(inds[0][1] + kernel_size)])
  box2 = np.sum(crop_image[(inds[1][0] - kernel_size):(inds[1][0] + kernel_size), (inds[1][1] - kernel_size):(inds[1][1] + kernel_size)])

  center_y = None
  center_x = None

  if box1 > box2:
    center_y = up_pos + inds[0][0]
    center_x = lt_pos + inds[0][1]
  else:
    center_y = up_pos + inds[1][0]
    center_x = lt_pos + inds[1][1]

  box_lt = center_x - coords_kernel_size[1] // 2
  box_rt = center_x + coords_kernel_size[1] // 2
  box_up = center_y - coords_kernel_size[0] // 2
  box_down = center_y + coords_kernel_size[0] // 2 + 15 # Because the middle of the box tends to be higher
  return box_up, box_down, box_lt, box_rt


def get_grade_coords(image):
  # HSV ------------------------------------------------------------------------
  # image = blur(image)
  # hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

  # red = np.uint8([[[255, 0, 0]]])
  # hsv_red = cv2.cvtColor(red, cv2.COLOR_RGB2HSV)
  # hue = hsv_red[0, 0, 0]

  # lower_red = np.array([329., 0.2, 50.])
  # upper_red = np.array([359., 1., 255.])
  # mask1 = cv2.inRange(hsv, lower_red, upper_red)

  # lower_red = np.array([0., 0.2, 50.])
  # upper_red = np.array([30., 1., 255.])
  # mask2 = cv2.inRange(hsv, lower_red, upper_red)

  # mask = np.clip(mask1.astype(np.int32) + mask2.astype(np.int32), a_min=None, a_max=255).astype(np.uint8)

  # res = cv2.bitwise_and(image, image, mask=mask)
  # ----------------------------------------------------------------------------

  lt = 345
  rt = 676
  up = 1711
  down = 1880

  grade_crop = image[up:down, lt:rt]

  max_x_argmax = np.argmax(grade_crop[:-20, :], axis=0)
  max_x = np.nonzero(max_x_argmax)[0][0]
  max_y = np.argmax(grade_crop[:, max_x])

  grade_lt = lt + max_x
  grade_rt = rt

  max_y_argmax = np.argmax(grade_crop, axis=1)
  max_y = np.nonzero(max_y_argmax)[0][0]
  max_x = np.argmax(grade_crop[max_y, :])

  grade_up = up + max_y
  grade_down = down

  return grade_up, grade_down, grade_lt, grade_rt


def get_box_number(image_original):
    pil_image = Image.fromarray(image_original.astype(np.uint8))
    image = np.array(pil_image.convert('L'))

    image[image < 127] = 0
    image[image >= 127] = 255
    image = np.pad(image, (20, 20), 'constant', constant_values=255)
    image = 255 - image

    pil_image = Image.fromarray(image)
    image = np.array(pil_image.resize((28, 28), Image.ANTIALIAS))

    number = predict_number_box(image)
    return number


def center_image(image):
    lt, up = 999, 999
    rt, down = -1, -1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < 127:
                continue

            lt = min(lt, j)
            rt = max(rt, j)
            up = min(up, i)
            down = max(down, i)

    rt += 1
    down += 1

    return image[up:down, lt:rt]


def get_grade_number(image_original):
    pil_image = Image.fromarray(image_original.astype(np.uint8))
    image = np.array(pil_image.convert('L'))

    image[image < 150] = 0
    image[image >= 150] = 255
    image = 255 - image
    image = dilate(image, 1)

    pil_image = Image.fromarray(image)
    ratio = 28. / image.shape[0]
    image = np.array(pil_image.resize((int(ratio * image.shape[1]), 28), Image.ANTIALIAS))

    window_size_width = 14
    image = np.pad(image, ((0, 0), (0, window_size_width)), 'constant', constant_values=0)

    res = []
    for pos in range(0, image.shape[1], 1):
        if pos + window_size_width - 1 >= image.shape[1]:
            break
        new_image = image[:, pos:pos + window_size_width - 1]
        new_image = center_image(new_image)
        new_image = np.pad(new_image, (5, 5), 'constant', constant_values=0)
        pil_image = Image.fromarray(new_image)
        new_image = np.array(pil_image.resize((28, 28), Image.ANTIALIAS))

        number, prob = predict_number_grade(new_image)
        res.append((number, prob))

    probs = [r[1] for r in res]
    number = [r[0] for r in res]

    output = []
    idx_sorted = np.argsort(probs)[::-1]
    idx_used = [0 for _ in range(len(idx_sorted))]
    for idx in idx_sorted:
        if idx_used[idx] == 1:
            continue

        output.append((idx, number[idx], probs[idx]))
        for i in np.arange(-10, 11):
            if idx + i < 0 or idx + i >= len(idx_sorted):
                continue
            pos = idx + i
            idx_used[pos] = 1

    sorted_output = sorted(output, key=lambda x: x[0])
    if len(sorted_output) < 2:
        sorted_output.append((-1, 0, -1))
        sorted_output.append((-1, 0, -1))
    elif len(sorted_output) < 3:
        sorted_output.append((-1, 0, -1))
    output_number = float('{}.{}{}'.format(sorted_output[0][1], sorted_output[1][1], sorted_output[2][1]))

    return output_number
