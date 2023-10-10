# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

np.set_printoptions(threshold=sys.maxsize)

from imglib import imglib

if __name__ == "__main__":
    img = cv2.imread('Lenna.jpg', cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # color

    my_img = imglib(img)
    gray_img = my_img.to_gray(my_img.process_img)
    my_img.show('original', gray_img)

    noise_img = my_img.apply_noise(gray_img)
    my_img.show('noise', noise_img)

    # homomorphic_img = my_img.homomorphic_filter(gray_img, c = 0.0001)
    # my_img.show('homorphic', homomorphic_img)
    # filter = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    filter = (1 / 25) * np.ones(shape = (5, 5))
    smoothing_img = my_img.conv(noise_img, filter=filter)

    my_img.show('smoothing', smoothing_img)

