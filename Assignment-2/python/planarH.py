import numpy as np
from matplotlib.pyplot import hist2d
import cv2

def computeH(x1, x2):
    """
    Compute the homography between two sets of points using the Direct Linear Transform (DLT) algorithm.

    Parameters:
    x1 (array): Coordinates in the first image. Shape should be (N, 2).
    x2 (array): Coordinates in the second image. Shape should be (N, 2).

    Returns:
    H2to1 (array): The 3x3 homography matrix that transforms points from the second image to the first image.
    """
    # Check if the number of points in both sets is equal
    assert len(x1) == len(x2), "The number of points in x1 and x2 must be the same."
    
    # Number of points
    n = len(x1)
    
    # Construct the matrix A, each point correspondence contributes two rows
    A = np.zeros((2 * n, 9))
    for i in range(n):
        A[2*i] = [-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0]]
        A[2*i + 1] = [0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1]]
    
    # Perform Singular Value Decomposition
    U, S, Vh = np.linalg.svd(A)
    
    # The homography is the last column of V (or the last row of Vh)
    H2to1 = Vh[-1, :].reshape(3, 3)
    
    return H2to1


def computeH_norm(x1, x2):
    """
    Compute the normalized homography between two sets of points.

    Parameters:
    x1 (array): Coordinates in the first image.
    x2 (array): Coordinates in the second image.

    Returns:
    H2to1 (array): The 3x3 normalized homography matrix.
    """
    assert len(x1) == len(x2), "Arrays must be the same length."

    # Compute the centroid of the points
    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    # Shift the centroid of the points to the origin
    x1_shifted = x1 - x1_centroid
    x2_shifted = x2 - x2_centroid

    # Normalize the points so that the average distance from the origin is equal to sqrt(2)
    dist1 = np.sqrt(np.sum(x1_shifted**2, axis=1))
    dist2 = np.sqrt(np.sum(x2_shifted**2, axis=1))
    avg_dist1 = np.mean(dist1)
    avg_dist2 = np.mean(dist2)
    scale1 = np.sqrt(2) / avg_dist1
    scale2 = np.sqrt(2) / avg_dist2
    x1_normalized = x1_shifted * scale1
    x2_normalized = x2_shifted * scale2

    # Construct the similarity transforms
    T1 = np.array([[scale1, 0, -x1_centroid[0] * scale1],
                   [0, scale1, -x1_centroid[1] * scale1],
                   [0, 0, 1]])
    T2 = np.array([[scale2, 0, -x2_centroid[0] * scale2],
                   [0, scale2, -x2_centroid[1] * scale2],
                   [0, 0, 1]])

    # Compute homography from the normalized coordinates
    H2to1 = computeH(x1_normalized, x2_normalized)

    # Denormalize the homography matrix
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    """
    Compute the best-fitting homography H using RANSAC.
    
    Parameters:
    locs1 (array): Array of points from the first image.
    locs2 (array): Array of points from the second image that correspond to locs1.
    opts (Namespace): Options containing max_iters and inlier_tol for RANSAC.
    
    Returns:
    bestH2to1 (array): The 3x3 homography matrix that best fits the points.
    inliers (array): Array indicating whether each point pair is an inlier.
    """
    max_iters = opts.max_iters  # The number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol  # The tolerance value for considering a point to be an inlier
    
    assert len(locs1) == len(locs2), "Point arrays must be the same length."
    
    # Swap columns in locs because they are in the form of [y, x] returned by matchPics
    x1 = locs1[:, [1, 0]]
    x2 = locs2[:, [1, 0]]
    
    # Initialize variables to keep track of the best homography and inliers
    bestH2to1 = None
    best_inliers = np.zeros(len(x1), dtype=bool)
    best_inliers_count = 0

    for iter in range(max_iters):
        # Randomly sample 4 pairs of points
        sample_indices = np.random.choice(len(x1), 4, replace=False)
        x1_samples = x1[sample_indices]
        x2_samples = x2[sample_indices]

        # Compute a homography from the samples
        H2to1 = computeH_norm(x1_samples, x2_samples)

        # Apply the homography to all points in x2
        x2_homo = np.concatenate((x2, np.ones((len(x2), 1))), axis=1)
        x2_transformed = (H2to1 @ x2_homo.T).T
        x2_transformed /= x2_transformed[:, 2:3]  # Avoid division by zero

        # Compute inliers where the transformed points are within the inlier tolerance
        inliers_binary = np.linalg.norm(x1 - x2_transformed[:, :2], axis=1) <= inlier_tol
        inliers_count = np.sum(inliers_binary)

        # Update the best homography if more inliers are found
        if inliers_count > best_inliers_count:
            bestH2to1 = H2to1
            best_inliers = inliers_binary
            best_inliers_count = inliers_count

        # If all points are inliers, we have found the best homography
        if inliers_count == len(x1):
            break

    return bestH2to1, best_inliers.astype(int)



def compositeH(H2to1, template, img):
    """
    Create a composite image by warping the template image onto the target image using a homography matrix.

    Args:
    H2to1 (numpy.ndarray): Homography matrix from the target image to the template.
    template (numpy.ndarray): Template image to be warped.
    img (numpy.ndarray): Target image on which the template will be overlaid.

    Returns:
    numpy.ndarray: The resulting composite image.
    """

    # Invert the homography matrix. Since the original homography is from the target image to the template,
    # we need to invert it to warp the template image onto the target image.
    H2to1_inv = np.linalg.inv(H2to1)

    # Create a mask of the same size as the template. This mask will be used to identify the region
    # in the warped template image that needs to be combined with the target image.
    mask = np.ones(template.shape)

    # Warp the mask using the inverted homography. This warps the mask to align with the perspective
    # of the target image, matching the transformation applied to the template.
    warped_m = cv2.warpPerspective(mask, H2to1_inv, (img.shape[1], img.shape[0]))

    # Warp the template image using the same inverted homography. This aligns the template
    # with the perspective and dimensions of the target image.
    warped_t = cv2.warpPerspective(template, H2to1_inv, (img.shape[1], img.shape[0]))

    # Combine the warped template with the target image. The mask is used to blend the template
    # onto the target image. Where the mask is true (region of the template), the template is used;
    # where the mask is false, the target image is retained.
    composite_img = warped_t + img * np.logical_not(warped_m)

    return composite_img