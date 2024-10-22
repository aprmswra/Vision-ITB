import numpy as np
import cv2
import sys
sys.path.append('./python')
from matplotlib import pyplot as plt
from MatchPics import matchPics
from planarH import computeH_ransac, compositeH
from opts import get_opts

# Konfigurasi untuk pencocokan fitur dan perhitungan homografi
opts = get_opts()
opts.sigma = 0.12
opts.ratio = 0.7
opts.inlier_tol = 1.4

# Memuat gambar kiri dan kanan dari panorama
left = cv2.imread('./data/crop_left.jpeg')
right = cv2.imread('./data/crop_right.jpeg')

# Cek apakah gambar berhasil dimuat
if left is None or right is None:
    print("Error loading images")
else:
    print("Left image shape:", left.shape)
    print("Right image shape:", right.shape)

# Melakukan pencocokan fitur antara gambar kiri dan kanan
matches, locs1, locs2 = matchPics(left, right, opts)

# Menghitung matriks homografi
H, _ = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]], opts)

# Mendapatkan dimensi dari gambar kiri dan kanan
hl, wl = left.shape[:2]
hr, wr = right.shape[:2]

# Warp gambar kanan untuk menyelaraskan dengan gambar kiri dan menjahitnya bersama-sama
res = cv2.warpPerspective(right, H, (wr + wl, max(hr, hl)))
res[:hl, :wl] = left

# Mengonversi hasilnya ke grayscale dan menerapkan thresholding untuk mengidentifikasi area hitam
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# Mencari kontur dan menghitung persegi pembatas untuk memotong area hitam
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

# Menyimpan gambar panorama yang telah dipotong
cv2.imwrite('./result/panaroma.jpg', res[y:y+h, x:x+w])
