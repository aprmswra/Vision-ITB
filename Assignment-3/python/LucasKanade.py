import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1,y1,x2,y2 = rect

    # Membuat interpolasi spline untuk gambar saat ini (It1) dan template (It)
    I = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    T = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    # Membuat array koordinat untuk wilayah template.
    X, Y = np.arange(x1,x2), np.arange(y1,y2)
    for iter in range(maxIters):
        # Menghitung koordinat yang distorsi berdasarkan vektor gerak p.
        Y_warp = Y+p[1]
        X_warp = X+p[0]

        # Menghitung gradien gambar di koordinat distorsi.
        Iy = I(Y_warp, X_warp, dx=1).flatten()
        Ix = I(Y_warp, X_warp, dy=1).flatten()

        # Menumpuk gradien untuk membentuk matriks Jacobian.
        J = np.vstack((Ix, Iy)).T

        # Menghitung selisih antara template dan wilayah yang distorsi.
        b = T(Y, X).flatten() - I(Y_warp, X_warp).flatten()

        # Menyelesaikan persamaan least squares untuk mendapatkan perubahan vektor gerak.
        dp = np.linalg.lstsq(J, b, rcond=None)[0]

        # update
        p += dp

        # Memeriksa kondisi konvergensi.
        if np.linalg.norm(dp) < threshold:
            break
    return p

