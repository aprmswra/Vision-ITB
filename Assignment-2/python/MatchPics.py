import numpy as np
import cv2
import skimage.color

from helper import briefMatch, computeBrief, corner_detection

def matchPics(I1, I2, opts):
    # Ekstrak parameter yang diperlukan dari opts
    ratio = opts.ratio  # Rasio untuk deskriptor fitur BRIEF
    sigma = opts.sigma  # Ambang batas untuk deteksi sudut menggunakan detektor fitur FAST

    # Konversi gambar ke skala abu-abu
    img1_gray = skimage.color.rgb2gray(I1)
    img2_gray = skimage.color.rgb2gray(I2)

    # Deteksi fitur di kedua gambar menggunakan nilai sigma yang ditentukan
    locs1 = corner_detection(img1_gray, sigma)
    locs2 = corner_detection(img2_gray, sigma)

    # Dapatkan deskriptor untuk lokasi fitur yang dihitung
    desc1, locs1 = computeBrief(img1_gray, locs1)
    desc2, locs2 = computeBrief(img2_gray, locs2)

    # Cocokkan fitur menggunakan deskriptor dan rasio yang ditentukan
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2
