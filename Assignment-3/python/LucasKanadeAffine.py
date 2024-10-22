import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))

    # Mendapatkan koordinat persegi panjang yang melacak objek.
    x1,y1,x2,y2 = rect

    # Memastikan bahwa kotak pembatas berada dalam batas gambar
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(It.shape[1], x2)
    y2 = min(It.shape[0], y2)

    # Membuat interpolasi spline untuk gambar saat ini (It1) dan template (It)
    I = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    T = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    # Membuat array koordinat untuk wilayah template.
    X, Y = np.arange(x1,x2), np.arange(y1,y2)

    # Mengambil nilai pixel pada wilayah template.
    Tx = T(Y, X)

    # Membuat grid koordinat untuk wilayah template.
    xx, yy = np.meshgrid(X, Y)
    for iter in range(maxIters):

        # Menghitung koordinat yang distorsi menggunakan transformasi affine.
        x_warp = xx*(1+p[0]) + yy*p[1] + p[2]
        y_warp = xx*p[3] + yy*(1+p[4]) + p[5]

        # Menghitung gradien gambar pada koordinat yang distorsi.
        Iy = I.ev(y_warp, x_warp, dx=1).flatten()
        Ix = I.ev(y_warp, x_warp, dy=1).flatten()

        # Meratakan koordinat yang distorsi.
        y = y_warp.flatten()
        x = x_warp.flatten()

        # Menumpuk gradien untuk membentuk matriks Jacobian.
        J = np.vstack((x*Ix, y*Ix, Ix, x*Iy, y*Iy, Iy)).T

        # Menghitung selisih antara template dan wilayah yang distorsi.
        b = (Tx - I.ev(y_warp, x_warp)).flatten()

        # Menyelesaikan persamaan least squares untuk mendapatkan perubahan parameter affine.
        dp = np.linalg.lstsq(J, b, rcond=None)[0].reshape(-1, 1)

        # Update parameters.
        p += dp

        # Memeriksa kondisi konvergensi.
        if np.linalg.norm(dp) < threshold:
            break

    # Membentuk matriks affine dari parameter yang diperbarui.
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
