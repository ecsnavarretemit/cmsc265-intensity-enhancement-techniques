# log-transformation.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np
from app import log_transform

# read the image file
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/morning/DSC_0380.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/noon/DSC_0389.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/afternoon/DSC_0400.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/evening/DSC_0412.JPG")
image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")

cv_image = cv2.imread(image)

height, width, _ = cv_image.shape
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# increment by 3
for constant in np.arange(1, 50, 3):
  # dont modify the original image
  gray_image_mod = gray_image.copy()

  # apply log transform by building look up table per pixel
  gray_image_mod = log_transform(gray_image_mod, constant)

  cv2.putText(gray_image_mod, "c={}".format(constant), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
  cv2.imshow('Result', np.hstack([gray_image, gray_image_mod]))
  cv2.waitKey(0)


