# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import math
import numpy as np

def im2double(im):
  info = np.iinfo(im.dtype) # Get the data type of the input image

  return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def adjust_gamma(img, gamma=1.0):
  # dont modify the original
  img = img.copy()

  # build a lookup table mapping the pixel values [0, 255] to
  # their adjusted gamma values
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

  # apply gamma correction using the lookup table
  return cv2.LUT(img, table)

def log_transform(img, constant_val=1):
  # dont modify the original
  img = img.copy()

  double_img = im2double(img)

  return constant_val * np.log(1 + double_img)

def equalize_histogram(img):
  # dont modify the original
  img = img.copy()

  img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

  # equalize the histogram of the Y channel
  img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

  # convert the YUV image back to RGB format
  return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


