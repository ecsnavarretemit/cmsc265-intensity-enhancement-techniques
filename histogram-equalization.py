# histogram-equalization.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from app import equalize_histogram_grayscale

# read the image file
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/morning/DSC_0380.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/noon/DSC_0389.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/afternoon/DSC_0400.JPG")
# image = os.path.join(os.getcwd(), "assets/images-small/outdoor/evening/DSC_0412.JPG")
image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")

cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
img_output = equalize_histogram_grayscale(cv_image.copy())

# resolve the output folder
save_folder = os.path.join(os.getcwd(), "out/histogram-equalization/indoor")
if not os.path.exists(save_folder):
  os.makedirs(save_folder)

# show the image and the graphs in one window
hist_original = cv2.calcHist([cv_image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([img_output], [0], None, [256], [0, 256])
plt.hist(cv_image.ravel(), 256, [0, 256], label="Input Image")
plt.hist(img_output.ravel(), 256, [0, 256], label="Output Image")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

# # save to file system
# plt.savefig(f"{save_folder}/chart.png")
# cv2.imwrite(f"{save_folder}/original.jpg", cv_image)
# cv2.imwrite(f"{save_folder}/equalized.jpg", img_output)

# # add labels to the images
# cv2.putText(cv_image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# cv2.putText(img_output, "Equalized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

plt.show()
# cv2.imshow('Result', np.hstack([cv_image, img_output]))
# cv2.waitKey(0)


