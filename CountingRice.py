#!/usr/bin/python3

# Copyright 2020 Sumandeep Banerjee, sumandeep.banerjee@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2 as cv
import numpy as np

# load input images for demonstration
input_rice = cv.imread("input/rice.png", cv.IMREAD_GRAYSCALE)

# local adaptive thresholding - computes local threshold based on given window size
output_adapthresh = cv.adaptiveThreshold (input_rice, 255.0,
		cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -20.0)
cv.imshow("Adaptive Thresholding", output_adapthresh)
#cv.imwrite('rice_adapthresh.png', output_adapthresh)

# morphologial erosion - cleaning up binary images
kernel = np.ones((5,5),np.uint8)
output_erosion = cv.erode(output_adapthresh, kernel)
cv.imshow("Morphological Erosion", output_erosion)
#cv.imwrite('rice_erosion.png', output_erosion)

# connected components - counts and marks number of distinct foreground objects
# apply connected components on clean binary image
label_image = output_erosion.copy()
label_count = 0
rows, cols = label_image.shape
for j in range(rows):
    for i in range(cols):
        pixel = label_image[j, i]
        if 255 == pixel:
            label_count += 1
            cv.floodFill(label_image, None, (i, j), label_count)

print("Number of foreground objects", label_count)
cv.imshow("Connected Components", label_image)
#cv.imwrite('rice_components.png', label_image)

# Contours - Computes polygonal contour boundary of foreground objects
# apply connected components on clean binary image
_, contours, _ = cv.findContours(output_erosion, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
output_contour = cv.cvtColor(input_rice, cv.COLOR_GRAY2BGR)
cv.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
print("Number of detected contours", len(contours))
cv.imshow("Contours", output_contour)
#cv.imwrite('rice_contours.png', output_contour)

# wait for key press
cv.waitKey(0)
cv.destroyAllWindows()
