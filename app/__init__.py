# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np

def adjust_gamma(img, gamma=1.0):
  # dont modify the original
  img = img.copy()

  # build a lookup table mapping the pixel values [0, 255] to
  # their adjusted gamma values
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

  # apply gamma correction using the lookup table
  return cv2.LUT(img, table)


