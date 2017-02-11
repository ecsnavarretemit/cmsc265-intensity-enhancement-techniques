# histogram-equalization.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np
from app import equalize_histogram

# read the image file
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/morning/DSC_0380.JPG")
image = os.path.join(os.getcwd(), "assets/images-small/outdoor/noon/DSC_0389.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/afternoon/DSC_0400.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/evening/DSC_0412.JPG")

cv_image = cv2.imread(image)
img_output = equalize_histogram(cv_image)

# add labels to the images
cv2.putText(cv_image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.putText(img_output, "Equalized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

cv2.imshow('Result', np.hstack([cv_image, img_output]))
cv2.waitKey(0)


