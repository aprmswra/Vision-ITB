import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
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
    Template = T(Y, X)

    # Membuat grid koordinat untuk wilayah template.
    xx, yy = np.meshgrid(X, Y)

    # Menghitung Matriks Jacobian.
    Ty = T.ev(yy, xx, dx=1).flatten()
    Tx = T.ev(yy, xx, dy=1).flatten()
    y = yy.flatten()
    x = xx.flatten()
    J = np.vstack((x*Tx, y*Tx, Tx, x*Ty, y*Ty, Ty)).T

    for iter in range(maxIters):
        # Menghitung koordinat yang distorsi menggunakan transformasi affine.
        x_warp = xx*(1+p[0]) + yy*p[1] + p[2]
        y_warp = xx*p[3] + yy*(1+p[4]) + p[5]

        # Menghitung selisih antara gambar saat ini dan template.
        b = (I.ev(y_warp, x_warp) - Template).flatten()

        # Menyelesaikan persamaan least squares untuk mendapatkan perubahan parameter affine.
        dp = np.linalg.lstsq(J, b, rcond=None)[0]
        
        # Memastikan bahwa p adalah array satu dimensi
        p = p.flatten()  

        # Membentuk matriks transformasi affine dari parameter saat ini.
        P = np.array([
            [1.0 + p[0], p[1], p[2]],
            [p[3], 1.0 + p[4], p[5]],
            [0, 0, 1]
        ])

        # Membentuk matriks transformasi affine dari perubahan parameter.
        DP = np.array([
            [1.0 + dp[0], dp[1], dp[2]],
            [dp[3], 1.0 + dp[4], dp[5]],
            [0, 0, 1]
        ])
        
        # Memperbarui parameter affine menggunakan invers.
        p = ((P @ np.linalg.inv(DP)) - np.eye(3))[:2, :].flatten()

        # Memeriksa kondisi konvergensi.
        if np.linalg.norm(dp) < threshold:
            break

    # Membentuk matriks affine dari parameter yang diperbarui.
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
