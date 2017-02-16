# gamma-correction.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np
from app import gamma_correction

# read the image file
image = os.path.join(os.getcwd(), "assets/images-small/outdoor/morning/DSC_0380.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/noon/DSC_0389.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/afternoon/DSC_0400.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/evening/DSC_0412.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")

cv_image = cv2.imread(image, 0)

# resolve the output folder
save_folder = os.path.join(os.getcwd(), "out/power-law-transform/outdoor/morning")
if not os.path.exists(save_folder):
  os.makedirs(save_folder)

# loop over various values of gamma
for gamma in np.arange(0.0, 4, 0.5):
	# ignore when gamma is 1 (there will be no change to the image)
  if gamma == 1:
    continue

	# apply gamma correction and show the images
  gamma = gamma if gamma > 0 else 0.1
  adjusted = gamma_correction(cv_image, gamma)
  cv2.putText(adjusted, "g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
  cv2.imshow("Images", np.hstack([cv_image, adjusted]))
  cv2.waitKey(0)

  # cv2.imwrite(f"{save_folder}/gamma-{gamma}.jpg", adjusted)


