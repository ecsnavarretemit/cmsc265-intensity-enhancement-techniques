#!/usr/bin/env python

# gamma-correction.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np
from app import adjust_gamma

# read the image file
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/morning/DSC_0380.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/noon/DSC_0389.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/afternoon/DSC_0400.JPG")
image = os.path.join(os.getcwd(), "assets/images-small/outdoor/evening/DSC_0412.JPG")

cv_image = cv2.imread(image)

# loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
	# ignore when gamma is 1 (there will be no change to the image)
  if gamma == 1:
    continue

	# apply gamma correction and show the images
  gamma = gamma if gamma > 0 else 0.1
  adjusted = adjust_gamma(cv_image, gamma)
  cv2.putText(adjusted, "g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
  cv2.imshow("Images", np.hstack([cv_image, adjusted]))
  cv2.waitKey(0)


