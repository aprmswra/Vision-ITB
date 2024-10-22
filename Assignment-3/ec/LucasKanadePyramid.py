import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanadePyramid(It, It1, rect, pyramid_levels=3):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   pyramid_levels: number of levels in the pyramid
    # Output:
    #   p: movement vector dx, dy
    
    # Generate image pyramids
    It_pyramid = [It]
    It1_pyramid = [It1]
    for i in range(1, pyramid_levels):
        It_pyramid.append(cv2.pyrDown(It_pyramid[i-1]))
        It1_pyramid.append(cv2.pyrDown(It1_pyramid[i-1]))

    # Setup threshold.
    p = np.zeros(2)
    threshold = 0.01875
    maxIters = 100

    # Process setiap level pyramid
    for level in range(pyramid_levels-1, -1, -1):
        # Menghitung skala berdasarkan level piramida saat ini.
        scale = 2**level

        # Menyesuaikan koordinat persegi panjang untuk level piramida saat ini.
        scaled_rect = [rect[0]/scale, rect[1]/scale, rect[2]/scale, rect[3]/scale]

        # Mendapatkan koordinat persegi panjang yang disesuaikan untuk level piramida saat ini.
        x1, y1, x2, y2 = scaled_rect

        # Membuat interpolasi spline untuk piramida dari gambar saat ini dan template
        I = RectBivariateSpline(np.arange(It1_pyramid[level].shape[0]), 
                                np.arange(It1_pyramid[level].shape[1]), 
                                It1_pyramid[level])
        T = RectBivariateSpline(np.arange(It_pyramid[level].shape[0]), 
                                np.arange(It_pyramid[level].shape[1]), 
                                It_pyramid[level])
        
        # Membuat array koordinat untuk wilayah template di level piramida saat ini.
        X, Y = np.arange(x1, x2), np.arange(y1, y2)

        for iter in range(maxIters):
            # Menghitung koordinat yang distorsi berdasarkan vektor gerak p dan skala saat ini.
            Y_warp = Y + p[1]/scale
            X_warp = X + p[0]/scale

            # Menghitung gradien gambar pada koordinat yang distorsi.
            Iy = I(Y_warp, X_warp, dx=1).flatten()
            Ix = I(Y_warp, X_warp, dy=1).flatten()

            # Menumpuk gradien untuk membentuk matriks Jacobian.
            J = np.vstack((Ix, Iy)).T

            # Menghitung selisih antara template dan wilayah yang distorsi.
            b = T(Y, X).flatten() - I(Y_warp, X_warp).flatten()

            # Menyelesaikan persamaan least squares untuk mendapatkan perubahan vektor gerak.
            dp = np.linalg.lstsq(J, b, rcond=None)[0]
            p += dp * scale

            # Memeriksa kondisi konvergensi.
            if np.linalg.norm(dp) < threshold:
                break

    return p
