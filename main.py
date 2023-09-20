import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

from imglib import imglib

if __name__ == "__main__":
    img = cv2.imread('Lenna.jpg', cv2.IMREAD_COLOR)
    my_img = imglib(img)
    my_img.info()
    gray_img = my_img.to_gray()
    histogram = my_img.to_histogram()
    # 按下任意鍵則關閉所有視窗
    cv2.imshow('Image', img)
    cv2.imshow('My Gray Image', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()