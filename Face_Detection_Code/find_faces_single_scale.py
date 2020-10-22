"""
 Princeton University, COS 429, Fall 2020
"""
import numpy as np
from hog36 import hog36
from logistic_prob import logistic_prob
from nms import nms


def find_faces_single_scale(img, stride, thresh, params, orientations, wrap180):
    """Find 36x36 faces in an image

    Args:
        img: an image
        stride: how far to move between locations at which the detector is run
        thresh: probability threshold for calling a detection a face
        params: trained face classifier parameters
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360

    Returns:
        outimg: copy of img with face locations marked
        probmap: probability map of face detections
    """
    windowsize = 36
    if stride > windowsize:
        stride = windowsize

    height, width = img.shape
    probmap = np.zeros([height, width])
    outimg = np.array(img)

    # Loop over windowsize x windowsize windows, advancing by stride
    # Fill in here
    hog_descriptor_size = 100 * orientations
    window_descriptor = np.zeros([hog_descriptor_size + 1])
    for i in range(0, height - windowsize, stride):
        for j in range(0, width - windowsize, stride):

            # Crop out a windowsize x windowsize window starting at (i,j)
            crop = img[i: i  + windowsize, j : j + windowsize]

            # Compute a HoG descriptor, and run the classifier
            window_descriptor[0] = 1
            window_descriptor[1:] = hog36(crop, orientations, wrap180)
            probability = logistic_prob(window_descriptor, params)



            # If probability of a face is below thresh, continue
            if probability < thresh:
                continue
            # Mark detection probability in probmap
            probmap[i, j] = probability

            if i > 0 and j > 0:
                for sub_i in range(i - stride, i):
                    for sub_j in range(j - stride, j):
                        crop = img[sub_i: sub_i + windowsize, sub_j: sub_j + windowsize]

                        # Compute a HoG descriptor, and run the classifier
                        window_descriptor[0] = 1
                        window_descriptor[1:] = hog36(crop, orientations, wrap180)
                        probability = logistic_prob(window_descriptor, params)
                        if probability > thresh:
                            probmap[sub_i, sub_j] = probability

    windows = np.full((height,width), windowsize, dtype=int)
    index_i, index_j, size = nms(probmap, windowsize, stride, windows)
    # Mark the face in outimg
    for i in range(len(index_i)):
        outimg[index_i[i], index_j[i]:index_j[i] + windowsize] = 255
        outimg[index_i[i] + windowsize - 1, index_j[i]:index_j[i] + windowsize] = 255
        outimg[index_i[i]:index_i[i] + windowsize, index_j[i]] = 255
        outimg[index_i[i]:index_i[i] + windowsize, index_j[i] + windowsize - 1] = 255

    return outimg, probmap

