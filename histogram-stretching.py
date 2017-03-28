# histogram-stretching.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from app import stretch_histogram_percentile

# read the image file
image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")

cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
stretched_image = stretch_histogram_percentile(cv_image.copy(), 5, 99)

# resolve the output folder
save_folder = os.path.join(os.getcwd(), "out/histogram-stretching/indoor")
if not os.path.exists(save_folder):
  os.makedirs(save_folder)

hist_original = cv2.calcHist([cv_image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([stretched_image], [0], None, [256], [0, 256])
plt.hist(cv_image.ravel(), 256, [0, 256], label="Input Image")
plt.hist(stretched_image.ravel(), 256, [0, 256], label="Output Image")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

# save to file system
plt.savefig(f"{save_folder}/chart.png")
cv2.imwrite(f"{save_folder}/original.jpg", cv_image)
cv2.imwrite(f"{save_folder}/stretched.jpg", stretched_image)

# plt.show()
# cv2.imshow('Result', np.hstack([cv_image, stretched_image]))
# cv2.waitKey(0)


