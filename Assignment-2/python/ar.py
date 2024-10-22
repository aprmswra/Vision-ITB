import numpy as np
import cv2
import multiprocessing
from matplotlib import pyplot as plt
from opts import get_opts
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from pathlib import Path

def cropBlackBar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    res = cv2.threshold(blur, 235, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5,5), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

def homographyWarp(args):
    i, opts, template, cv_cover, cv_book = args
    template = cv2.resize(template, (cv_cover.shape[1], cv_cover.shape[0]))
    matches, locs1, locs2 = matchPics(cv_cover, cv_book, opts)
    bestH2to1, _ = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]], opts)
    composite_img = compositeH(bestH2to1, template, cv_book)
    print(f"Frame {i} Processed")
    return composite_img

if __name__ == '__main__':
    opts = get_opts()
    opts.sigma = 0.12
    opts.ratio = 0.7
    opts.inlier_tol = 1.4
    n_worker = multiprocessing.cpu_count()

    ar_source_mov_path = 'D:/ITB/Kuliah/Semester 3/IF6083 - Vision/Tugas/Assignment - 3/supplements_fahmi/data/ar_source.mov'
    ar_source_npy_path = Path('D:/ITB/Kuliah/Semester 3/IF6083 - Vision/Tugas/Assignment - 3/supplements_fahmi/ar_source.npy')
    cv_cover = cv2.imread('D:/ITB/Kuliah/Semester 3/IF6083 - Vision/Tugas/Assignment - 3/supplements_fahmi/data/cv_cover.jpg')
    ar_book_mov_path = 'D:/ITB/Kuliah/Semester 3/IF6083 - Vision/Tugas/Assignment - 3/supplements_fahmi/data/book.mov'
    ar_book_npy_path = Path('D:/ITB/Kuliah/Semester 3/IF6083 - Vision/Tugas/Assignment - 3/supplements_fahmi/book_source.npy')

    ar_source_load = loadVid(ar_source_mov_path)
    ar_book_load = loadVid(ar_book_mov_path)
    np.save(ar_source_npy_path, ar_source_load)
    np.save(ar_book_npy_path, ar_book_load)

    ar_source = np.load(ar_source_npy_path, allow_pickle=True)
    ar_book = np.load(ar_book_npy_path, allow_pickle=True)

    frame0 = ar_source[0]
    x, y, w, h = cropBlackBar(frame0)
    frame0 = frame0[y:y+h, x:x+w]
    H, W = frame0.shape[:2]
    width = cv_cover.shape[1] * H / cv_cover.shape[0]
    wStart, wEnd = np.round([W/2 - width/2, W/2 + width/2]).astype(int)
    frame0 = frame0[:, wStart:wEnd]
    new_source = np.array([f[y:y+h, x:x+w][:, wStart:wEnd] for f in ar_source])

    args = [(i, opts, new_source[i], cv_cover, ar_book[i]) for i in range(len(new_source))]

    with multiprocessing.Pool(processes=n_worker) as p:
        ar = p.map(homographyWarp, args)

    ar = np.array(ar)
    writer = cv2.VideoWriter('./result/ar.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (ar.shape[2], ar.shape[1]))

    for i, f in enumerate(ar):
        writer.write(f)
        plt.figure()
        plt.axis('off')
        plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        plt.savefig(f'./result/frame_{i}.png')
        plt.close()

    writer.release()
