import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW2: Homography')
    parser.add_argument('--sigma', type=float, default=0.65,
                        help='Threshold for corner detection using FAST feature detector')
    
    parser.add_argument('--ratio', type=float, default=1.5,
                        help='Ratio for BRIEF feature descriptor')

    parser.add_argument('--max_iters', type=int, default=500,
                        help='The number of iterations to run RANSAC for')
    
    parser.add_argument('--inlier_tol', type=float, default=2.0,
                        help='The tolerance value for considering a point to be an inlier')

    opts, _ = parser.parse_known_args()

    return opts