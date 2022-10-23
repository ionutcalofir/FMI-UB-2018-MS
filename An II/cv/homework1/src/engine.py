import numpy as np
import matplotlib.pyplot as plt

import utils.util as utils

def get_image_info(query_path, template_path, verbose=0):
    query_original = utils.read_image(query_path)
    query_original_rgb = utils.read_image(query_path, rgb=True)
    template_original = utils.read_image(template_path)

    query = utils.erode(query_original, kernel_size=(5, 5))
    template = utils.erode(template_original, kernel_size=(5, 5))

    query = utils.blur(query, kernel_size=(7, 7))
    template = utils.blur(template, kernel_size=(7, 7))

    query_frames, query_descriptors = utils.get_sift_features(query)
    template_frames, template_descriptors = utils.get_sift_features(template)

    flann = utils.get_flann_kdtree()
    matches = flann.knnMatch(query_descriptors, template_descriptors, k=2)

    good_matches = utils.preprocess_matches(matches)

    query_pts = np.array([[query_frames[m.queryIdx][1], query_frames[m.queryIdx][0]] for m in good_matches]) # (X, Y)
    template_pts = np.array([[template_frames[m.trainIdx][1], template_frames[m.trainIdx][0]] for m in good_matches])

    query_scale = np.array([query_frames[m.queryIdx][2] for m in good_matches])
    template_scale = np.array([template_frames[m.trainIdx][2] for m in good_matches])

    M, mask = utils.get_homography(query_pts, template_pts)

    homography_query_pts = query_pts[mask[:, 0] == 1]
    homography_template_pts = template_pts[mask[:, 0] == 1]
    homography_query_scale = query_scale[mask[:, 0] == 1]
    homography_template_scale = template_scale[mask[:, 0] == 1]

    if verbose:
        fig = plt.figure('Query')
        plt.imshow(query_original, cmap='gray')
        plt.scatter(y=homography_query_pts[:, 1], x=homography_query_pts[:, 0],
                    facecolors='none', edgecolors='g',
                    s=homography_query_scale)
        plt.savefig('logs/' + 'Query' + '.png', bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure('Template')
        plt.imshow(template_original, cmap='gray')
        plt.scatter(y=homography_template_pts[:, 1], x=homography_template_pts[:, 0],
                    facecolors='none', edgecolors='g',
                    s=homography_template_scale)
        plt.savefig('logs/' + 'Template' + '.png', bbox_inches='tight')
        plt.close(fig)

        # plt.show()

    output_shape = np.array([[0, 0], [0., query_original.shape[0]], [query_original.shape[1], 0.], [query_original.shape[1], query_original.shape[0]]], dtype=np.float32)
    transform_output_shape = utils.apply_perspective_transform(output_shape, M)

    translate_x = np.min(transform_output_shape[:, 0])
    translate_x = abs(min(0., translate_x))
    translate_y = np.min(transform_output_shape[:, 1])
    translate_y = abs(min(0., translate_y))

    translate_transform =  np.array([[1., 0., translate_x], [0., 1., translate_y], [0., 0., 1.]], dtype=np.float32)
    translate_M = np.matmul(translate_transform, M)

    transform_output_shape_translate = utils.apply_perspective_transform(output_shape, translate_M)
    max_x = np.max(transform_output_shape_translate[:, 0])
    max_y = np.max(transform_output_shape_translate[:, 1])

    query_warped = utils.warp_perspective(query_original, translate_M, max_x, max_y)
    query_warped_rgb = utils.warp_perspective(query_original_rgb, translate_M, max_x, max_y)
    if verbose:
        fig = plt.figure('Query Warped')
        plt.imshow(query_warped, cmap='gray')
        plt.savefig('logs/' + 'Query Warped' + '.png', bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    transformed_homography_query_pts = utils.apply_perspective_transform(homography_query_pts, translate_M)

    template_mean_pt = np.mean(homography_template_pts, axis=0)
    query_mean_pt = np.mean(transformed_homography_query_pts, axis=0)

    up = -1. * template_mean_pt[1]
    down = template_original.shape[0] - template_mean_pt[1]
    left = -1. * template_mean_pt[0]
    right = template_original.shape[1] - template_mean_pt[0]
    query_final = query_warped[int(query_mean_pt[1] + up):int(query_mean_pt[1] + down), int(query_mean_pt[0] + left):int(query_mean_pt[0] + right)]
    query_final_rgb = query_warped_rgb[int(query_mean_pt[1] + up):int(query_mean_pt[1] + down), int(query_mean_pt[0] + left):int(query_mean_pt[0] + right), :]
    if verbose:
        fig = plt.figure('Query Final')
        plt.imshow(query_final, cmap='gray')
        plt.savefig('logs/' + 'Query Final' + '.png', bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    return query_final, query_final_rgb

def get_image_data(image, image_rgb, verbose=0):
    sobel_x = utils.sobel_x()
    sobel_y = utils.sobel_y()

    img = utils.blur(image, kernel_size=(5, 5))
    gx = utils.apply_filter(img, sobel_x)
    gy = utils.apply_filter(img, sobel_y)

    magnitude = np.sqrt(gx * gx + gy * gy)

    magnitude[magnitude < 50] = 0.
    magnitude[magnitude >= 50] = 255.

    magnitude = utils.dilate(magnitude, iterations=2)

    if verbose:
        fig = plt.figure('Magnitude')
        plt.imshow(magnitude, cmap='gray')
        plt.savefig('logs/' + 'Magnitude' + '.png', bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    box_filter = utils.get_box_filter()
    header_filter = utils.get_box_filter(kernel_size=(227, 1052))
    x_box_filter = utils.get_box_filter(kernel_size=(105,150))

    obox = utils.apply_filter(magnitude, box_filter)
    oheader = utils.apply_filter(magnitude, header_filter)
    ox_box = utils.apply_filter(magnitude, x_box_filter)

    obox[obox < 7.5e4] = 0.
    oheader[oheader < 5e5] = 0.
    ox_box[ox_box < 8.5e4] = 0.

    obox = utils.apply_nms(obox, verbose=verbose, fig_name='nms box')
    oheader = utils.apply_nms(oheader, kernel_size=(20, 300), verbose=verbose, fig_name='nms header')
    ox_box = utils.apply_nms(ox_box, verbose=verbose, fig_name='nms x_box')

    try:
      col1_header_pos, col2_header_pos = utils.get_header_pos(oheader)

      table1_x = utils.get_xs(magnitude, ox_box, col1_header_pos)
      table2_x = utils.get_xs(magnitude, ox_box, col2_header_pos)

      box_up, box_down, box_lt, box_rt = utils.get_box_coords(magnitude, obox)
      grade_up, grade_down, grade_lt, grade_rt = utils.get_grade_coords(magnitude)
      # grade_hsv = utils.get_grade_coords(image_rgb)
    except:
      raise Exception('Image was not propetly extracted. Check logs!')

    return table1_x, table2_x, (box_up, box_down, box_lt, box_rt), (grade_up, grade_down, grade_lt, grade_rt)
