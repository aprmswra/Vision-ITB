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

# Impor fungsi-fungsi yang diperlukan
# ... (Dengan asumsi fungsi-fungsi seperti get_opts, matchPics, computeH_ransac, compositeH, dll., sudah didefinisikan di tempat lain)

def read_and_resize_images(coverpath, deskpath, newcoverpath):
    originalcover = cv2.imread(coverpath)
    deskimage = cv2.imread(deskpath)
    newcover = cv2.imread(newcoverpath)

    # Ubah ukuran gambar sampul baru agar sesuai dengan ukuran sampul asli
    newcoverresized = cv2.resize(newcover, (originalcover.shape[1], originalcover.shape[0]))
    return originalcover, deskimage, newcoverresized

def process_images(cvcover, cvdesk, hpcover, opts, maxiters, tolvalues):
    matches, locs1, locs2 = matchPics(cvcover, cvdesk, opts)
    # plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

    for max_iter in maxiters:
        for tol in tolvalues:
            opts.max_iters = max_iter
            opts.inlier_tol = tol
            optimalhomography, inliers = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
            compositeimg = compositeH(optimalhomography, hpcover, cvdesk)

            plt.figure()
            plt.axis('off')
            plt.imshow(cv2.cvtColor(compositeimg, cv2.COLOR_BGR2RGB))
            plt.savefig(f'./result/pic_{max_iter}_{tol}.png')

# Inisialisasi opsi (parameter untuk berbagai fungsi)
opts = get_opts()

# Baca dan ubah ukuran gambar
cvcover, cvdesk, hpcover = read_and_resize_images('./data/cv_cover.jpg', './data/cv_desk.png', './data/hp_cover.jpg')

# Kisaran hiperparameter
maxiterations = [1250, 2500, 5000]
tolerances = [2, 10, 50]

# Proses gambar dengan hiperparameter RANSAC yang berbeda
process_images(cvcover, cvdesk, hpcover, opts, maxiterations, tolerances)
