# log-transformation.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.1

import os
import cv2
import numpy as np
from app import log_transform

# read the image file
image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")

cv_image = cv2.imread(image)

height, width, _ = cv_image.shape
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# resolve the output folder
save_folder = os.path.join(os.getcwd(), "out/log-transform/indoor")
if not os.path.exists(save_folder):
  os.makedirs(save_folder)

# increment by 3
for constant in np.arange(1, 10):
  # dont modify the original image
  gray_image_copy = gray_image.copy()

  # # apply log transform by building look up table per pixel
  gray_image_mod = log_transform(gray_image_copy, constant)

  cv2.putText(gray_image_mod, "c={}".format(constant), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
  cv2.imshow('Log Result', np.hstack([(gray_image / 255), gray_image_mod]))
  cv2.waitKey(0)

  # cv2.imwrite(f"{save_folder}/constant{constant}.jpg", (gray_image_mod * 255))


