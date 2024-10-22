import numpy as np
import cv2
import skimage.io 
import skimage.color
from matplotlib import pyplot as plt

from opts import get_opts
from MatchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from helper import plotMatches

# Import necessary functions
# ... (Assuming functions like get_opts, matchPics, computeH_ransac, compositeH, etc., are defined elsewhere)

def read_and_resize_images(cover_path, desk_path, new_cover_path):
    """
    Read and resize images for processing.

    Args:
    cover_path (str): Path to the original book cover.
    desk_path (str): Path to the desk image with the book on it.
    new_cover_path (str): Path to the new cover to be mapped on the book.

    Returns:
    Tuple of cv2 images: (original cover, desk image, resized new cover)
    """
    original_cover = cv2.imread(cover_path)
    desk_image = cv2.imread(desk_path)
    new_cover = cv2.imread(new_cover_path)

    # Resize the new cover image to match the original cover size
    new_cover_resized = cv2.resize(new_cover, (original_cover.shape[1], original_cover.shape[0]))
    return original_cover, desk_image, new_cover_resized

def process_images(cv_cover, cv_desk, hp_cover, opts, max_iters, tol_values):
    """
    Process the images by matching features, computing homography, and creating composite images.
    Iterate over different hyperparameters for RANSAC algorithm.
    """
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    # plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

    for max_iter in max_iters:
        for tol in tol_values:
            opts.max_iters = max_iter
            opts.inlier_tol = tol
            bestH2to1, inliers = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
            composite_img = compositeH(bestH2to1, hp_cover, cv_desk)

            plt.figure()
            plt.axis('off')
            plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
            plt.savefig(f'./result/pic_{max_iter}_{tol}.png')

# Initialize options (parameters for various functions)
opts = get_opts()

# Read and resize images
cv_cover, cv_desk, hp_cover = read_and_resize_images('./data/cv_cover.jpg', './data/cv_desk.png', './data/hp_cover.jpg')

# Hyperparameter ranges
max_iterations = [1250, 2500, 5000]
tolerances = [2, 10, 50]

# Process images with different RANSAC hyperparameters
process_images(cv_cover, cv_desk, hp_cover, opts, max_iterations, tolerances)
