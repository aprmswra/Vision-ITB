import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

# Mendefinisikan fungsi LucasKanadeRobust dengan input gambar template (It),
# gambar saat ini (It1), dan posisi objek saat ini (rect)
def LucasKanadeRobust(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    # Output:
    #   p: movement vector dx, dy
    
    # Menetapkan nilai ambang batas untuk perubahan vektor gerak
    threshold = 0.01875

    # Menetapkan jumlah iterasi maksimum untuk proses iteratif
    maxIters = 100

    # Membuat vektor gerak awal dengan nilai nol
    p = np.zeros(2)     

    # Mendapatkan koordinat persegi panjang yang melacak objek     
    x1, y1, x2, y2 = rect

    # Membuat interpolasi spline untuk gambar template (It) dan gambar saat ini (It1)
    I = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    T = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    # Membuat array koordinat untuk wilayah template
    X, Y = np.arange(x1, x2), np.arange(y1, y2)

    # Mengambil nilai pixel pada wilayah template
    template_region = T(Y, X).flatten()

    # Menghitung kecerahan rata-rata dari wilayah template
    avg_brightness_template = np.mean(template_region)

    for iter in range(maxIters):
        # Menghitung koordinat yang distorsi berdasarkan vektor gerak p
        Y_warp = Y + p[1]
        X_warp = X + p[0]

        # Menghitung gradien gambar di koordinat distorsi
        Iy = I(Y_warp, X_warp, dx=1).flatten()
        Ix = I(Y_warp, X_warp, dy=1).flatten()

        # Menumpuk gradien untuk membentuk matriks Jacobian
        J = np.vstack((Ix, Iy)).T

        # Mengambil nilai pixel pada wilayah yang telah distorsi
        warped_region = I(Y_warp, X_warp).flatten()

        # Menghitung kecerahan rata-rata dari wilayah yang distorsi
        avg_brightness_warped = np.mean(warped_region)

        # Menyesuaikan skala kecerahan wilayah yang distorsi
        if avg_brightness_warped != 0:
            brightness_scale = avg_brightness_template / avg_brightness_warped
            warped_region *= brightness_scale

        # Menghitung perbedaan antara wilayah template dan wilayah yang distorsi
        b = template_region - warped_region

        # Menghitung bobot berdasarkan estimator M yang dipilih
        weights = compute_weights(b) 

        # Membuat matriks diagonal dari bobot
        Lambda = np.diag(weights)

        # Menyiapkan sistem persamaan terbobot untuk least squares
        A = J.T @ Lambda @ J
        b_weighted = J.T @ Lambda @ b

        # Menghitung perubahan pada vektor gerak menggunakan least squares
        dp = np.linalg.lstsq(A, b_weighted, rcond=None)[0]

        # Update parameters
        p += dp

        # Memeriksa kondisi konvergensi
        if np.linalg.norm(dp) < threshold:
            break

    return p

# Mendefinisikan fungsi untuk menghitung bobot berdasarkan estimator M robust
def compute_weights(residuals):
    # Menetapkan nilai konstanta untuk estimator Huber
    k = 1.345 

    # Mengembalikan array bobot menggunakan fungsi pembobotan Huber
    return np.array([huber_weight(r, k) for r in residuals])

# Mendefinisikan fungsi pembobotan Huber
def huber_weight(r, k):
    if abs(r) <= k:
        return 1
    else:
        return k / abs(r)
