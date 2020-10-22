"""
 Princeton University, COS 429, Fall 2020
"""
from logistic_prob import logistic_prob
from hog36 import hog36
import numpy as np
import cv2
from nms import nms

def find_faces(img, stride, thresh, params, orientations, wrap180):
    """Find faces in an image

    Args:
        img: an image
        stride: how far to move between locations at which the detector is run,
            at the finest (36x36) scale.  This is effectively scaled up for
            larger windows.
        thresh: probability threshold for calling a detection a face
        params: trained face classifier parameters
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360

    Returns:
        outimg: copy of img with face locations marked
    """
    # Fill in here
    windowsize = 36
    original_window_size = 36
    if stride > windowsize:
        stride = windowsize
    stride_ratio = stride/windowsize
    original_stride = stride
    height, width = img.shape
    probmap = np.zeros([height, width])
    windows = np.zeros([height, width])
    outimg = np.array(img)

    # Loop over windowsize x windowsize windows, advancing by stride
    # Fill in here
    hog_descriptor_size = 100 * orientations
    window_descriptor = np.zeros([hog_descriptor_size + 1])
    while windowsize < min(height,width):
        for i in range(0, height - windowsize, stride):
            for j in range(0, width - windowsize, stride):

                # Crop out a windowsize x windowsize window starting at (i,j)
                crop = img[i: i + windowsize, j: j + windowsize]
                crop = cv2.resize(crop, (original_window_size, original_window_size))

                # Compute a HoG descriptor, and run the classifier
                window_descriptor[0] = 1
                window_descriptor[1:] = hog36(crop, orientations, wrap180)
                probability = logistic_prob(window_descriptor, params)

                # If probability of a face is below thresh, continue
                if probability < thresh:
                    continue

                # Mark detection probability in probmap
                if np.max(probmap[i, j]) < probability:
                    probmap[i, j] = probability
                    windows[i, j] = windowsize

        windowsize = int(windowsize * 1.2)
        stride = int(stride_ratio*windowsize)

    index_i, index_j, size = nms(probmap,original_window_size, original_stride, windows)

    for i in range(len(index_i)):
        # Mark the face in outimg
        outimg[index_i[i], index_j[i]:index_j[i] + size[i]] = 255
        outimg[index_i[i] + size[i] - 1, index_j[i]:index_j[i] + size[i]] = 255
        outimg[index_i[i]:index_i[i] + size[i], index_j[i]] = 255
        outimg[index_i[i]:index_i[i] + size[i], index_j[i] + size[i]- 1] = 255
    return outimg
