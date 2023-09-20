# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

np.set_printoptions(threshold=sys.maxsize)

from imglib import imglib

if __name__ == "__main__":
    img = cv2.imread('Lenna.jpg', cv2.IMREAD_COLOR)
    my_img = imglib(img)
    gray_img = my_img.to_gray()
    histogram = my_img.to_histogram()

    my_img.info()
    my_img.show('gray', gray_img)
    