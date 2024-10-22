import numpy as np
from matplotlib.pyplot import hist2d
import cv2

def computeH(x1, x2):
    # Periksa apakah jumlah titik pada kedua set sama
    assert len(x1) == len(x2), "Jumlah titik pada x1 dan x2 harus sama."

    # Jumlah titik
    N = len(x1)
    
    # Membangun matriks A, setiap korespondensi titik memberikan dua baris
    A = np.zeros((2 * N, 9))
    for i in range(N):
        A[2*i] = [-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0]]
        A[2*i + 1] = [0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1]]
    
    # Melakukan Dekomposisi Nilai Singular (SVD)
    _, _, Vh = np.linalg.svd(A)
    
    # Homografi diperoleh dari kolom terakhir V (atau baris terakhir Vh)
    homography = Vh[-1, :].reshape(3, 3)
    
    return homography



def computeH_norm(x1, x2):
    # Pastikan bahwa panjang kedua array sama
    assert len(x1) == len(x2), "Panjang kedua array harus sama."

    # Hitung centroid (titik tengah) dari kedua set titik
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)

    # Geser centroid kedua set titik ke asal (origin)
    shifted1 = x1 - centroid1
    shifted2 = x2 - centroid2

    # Normalisasi titik-titik sehingga jarak rata-rata dari asal adalah sama dengan akar kuadrat dari 2
    distance1 = np.sqrt(np.sum(shifted1**2, axis=1))
    distance2 = np.sqrt(np.sum(shifted2**2, axis=1))
    avgDist1 = np.mean(distance1)
    avgDist2 = np.mean(distance2)
    scale1 = np.sqrt(2) / avgDist1
    scale2 = np.sqrt(2) / avgDist2
    normalized1 = shifted1 * scale1
    normalized2 = shifted2 * scale2

    # Membangun similarity transform untuk kedua set titik
    transform1 = np.array([[scale1, 0, -centroid1[0] * scale1],
                           [0, scale1, -centroid1[1] * scale1],
                           [0, 0, 1]])
    transform2 = np.array([[scale2, 0, -centroid2[0] * scale2],
                           [0, scale2, -centroid2[1] * scale2],
                           [0, 0, 1]])

    # Hitung homografi dari koordinat yang telah dinormalisasi
    homography = computeH(normalized1, normalized2)

    # Denormalisasi matriks homografi
    homography = np.linalg.inv(transform1) @ homography @ transform2

    return homography


def computeH_ransac(locs1, locs2, opts):
    # Jumlah iterasi maksimum untuk menjalankan RANSAC
    max_iters = opts.max_iters
    # Nilai toleransi untuk menentukan titik sebagai inlier
    inlier_tol = opts.inlier_tol
    
    # Pastikan jumlah titik dalam kedua set sama
    assert len(locs1) == len(locs2), "Panjang array titik harus sama."

    # Tukar kolom di locs karena mereka dalam format [y, x] yang dikembalikan oleh matchPics
    x1 = locs1[:, [1, 0]]
    x2 = locs2[:, [1, 0]]
    
    # Inisialisasi variabel untuk melacak homografi terbaik dan inlier
    optimalhomography = None
    optimal_inliers = np.zeros(len(x1), dtype=bool)
    optimal_inliers_count = 0

    for iter in range(max_iters):
        # Pilih secara acak 4 pasang titik
        sample_indices = np.random.choice(len(x1), 4, replace=False)
        x1samples = x1[sample_indices]
        x2samples = x2[sample_indices]

        # Hitung homografi dari sampel
        homography = computeH_norm(x1samples, x2samples)

        # Terapkan homografi ke semua titik di x2
        x2homo = np.concatenate((x2, np.ones((len(x2), 1))), axis=1)
        x2transformed = (homography @ x2homo.T).T
        x2transformed /= x2transformed[:, 2:3]  # Hindari pembagian dengan nol

        # Hitung inlier di mana titik yang ditransformasi berada dalam toleransi inlier
        inliers_binary = np.linalg.norm(x1 - x2transformed[:, :2], axis=1) <= inlier_tol
        inliers_count = np.sum(inliers_binary)

        # Perbarui homografi terbaik jika menemukan lebih banyak inlier
        if inliers_count > optimal_inliers_count:
            optimalhomography = homography
            optimal_inliers = inliers_binary
            optimal_inliers_count = inliers_count

        # Jika semua titik adalah inlier, maka homografi terbaik telah ditemukan
        if inliers_count == len(x1):
            break

    return optimalhomography, optimal_inliers.astype(int)

def compositeH(H2to1, template, img):
    # Balikkan matriks homografi. Karena homografi asli dari gambar target ke template,
    # kita perlu membalikkannya untuk memproyeksikan gambar template ke atas gambar target.
    H2to1_inv = np.linalg.inv(H2to1)

    # Buat masker dengan ukuran yang sama dengan template. Masker ini akan digunakan untuk mengidentifikasi wilayah
    # dalam gambar template yang diproyeksikan yang perlu digabungkan dengan gambar target.
    mask = np.ones(template.shape)

    # Proyeksikan masker menggunakan homografi yang dibalik. Ini memproyeksikan masker agar selaras dengan perspektif
    # gambar target, sesuai dengan transformasi yang diterapkan pada template.
    warped_m = cv2.warpPerspective(mask, H2to1_inv, (img.shape[1], img.shape[0]))

    # Proyeksikan gambar template menggunakan homografi yang sama yang dibalik. Ini menyesuaikan template
    # dengan perspektif dan dimensi gambar target.
    warped_t = cv2.warpPerspective(template, H2to1_inv, (img.shape[1], img.shape[0]))

    # Gabungkan template yang telah diproyeksikan dengan gambar target. Masker digunakan untuk mencampur template
    # ke dalam gambar target. Di mana masker bernilai benar (wilayah template), template digunakan;
    # di mana masker bernilai salah, gambar target yang dipertahankan.
    composite_img = warped_t + img * np.logical_not(warped_m)

    return composite_img
