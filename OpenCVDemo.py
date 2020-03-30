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

print("Using OpenCV version", cv.__version__)


# load input images for demonstration
input_color = cv.imread("input/lena.png", cv.IMREAD_UNCHANGED)
input_grey = cv.imread("input/lena_grey.png", cv.IMREAD_UNCHANGED)
input_uniform = cv.imread("input/lena_uniform.png", cv.IMREAD_UNCHANGED)
input_random = cv.imread("input/lena_random.png", cv.IMREAD_UNCHANGED)
input_grey_faded = cv.imread("input/lena_grey_faded.png", cv.IMREAD_UNCHANGED)
input_rice = cv.imread("input/rice.png", cv.IMREAD_UNCHANGED)
input_lungCT = cv.imread("input/lungCT.png", cv.IMREAD_COLOR)


# image inversion - computes image negative
# invert a 3-channel BGR color image
output_inv = cv.bitwise_not(input_color)
cv.imshow("Image Inversion", output_inv)
cv.imwrite('output/lena_inv.png', output_inv)


# gaussian smoothing - reduces uniform noise
# apply gaussian blurr to BGR image with added uniform noise
output_gauss = cv.GaussianBlur(input_uniform, (5, 5), 0)
cv.imshow("Gaussian Smoothing", output_gauss)
cv.imwrite('output/lena_gauss.png', output_gauss)


# median filtering - reduces random noise
# apply median blurr to BGR image with added random noise
output_median = cv.medianBlur(input_random, 5)
cv.imshow("Median Filtering", output_median)
cv.imwrite('output/lena_median.png', output_median)


# laplacian operator - detects 2nd order derivative
# apply laplacian to greyscale image
output_laplacian = cv.convertScaleAbs(cv.Laplacian(input_grey, cv.CV_32F))
cv.imshow("Laplacian", output_laplacian)
cv.imwrite('output/lena_laplace.png', output_laplacian)


# laplacian of gaussian - detects edges using 2nd order derivative
# apply laplacian of gaussian to greyscale image
output_log = cv.convertScaleAbs(cv.Laplacian(cv.GaussianBlur(input_grey, (5, 5), 0), cv.CV_32F))
cv.imshow("Laplacian of Gaussian (LoG)", output_log)
cv.imwrite('output/lena_LoG.png', output_log)


# sobel filter - detects 1st order derivative (gradient)
# apply sobel filter along vertical and horizontal axis
output_sobelx = cv.convertScaleAbs(cv.Sobel(input_grey, cv.CV_32F, 1, 0, ksize = 3))
output_sobely = cv.convertScaleAbs(cv.Sobel(input_grey, cv.CV_32F, 0, 1, ksize = 3))
cv.imshow("Sobel Filter Vertical", output_sobelx)
cv.imshow("Sobel Filter Horizontal", output_sobely)
cv.imwrite('output/lena_sobelx.png', output_sobelx)
cv.imwrite('output/lena_sobely.png', output_sobely)


# canny edge detection - detects edges via edge-linking of weak and strong edges
# apply canny filter to greyscale image
output_canny = cv.Canny(input_grey, 50, 200) # 200 threshold for selecting strong edges
                                             # 50 threshold for weak edges linking strong edges
cv.imshow("Canny Edge Detection", output_canny)
cv.imwrite('output/lena_canny.png', output_canny)


# histogram equalization - enhances contrast
# apply histogram equalization to faded greyscale image
output_histeq = cv.equalizeHist(input_grey_faded)
cv.imshow("Histogram Equalization", output_histeq)
cv.imwrite('output/lena_histeq.png', output_histeq)


# binary thresholding (fixed) - segments image into two levels
thresh, output_binthresh = cv.threshold(input_rice, 127, 255, cv.THRESH_BINARY)
print("Fixed threshold", thresh)
cv.imshow("Binary Threshold (fixed)", output_binthresh)
cv.imwrite('output/rice_binthresh.png', output_binthresh)


# binary thresholding (otsu) - segments image into two levels
thresh, output_otsuthresh = cv.threshold(input_rice, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
print("Otsu threshold", thresh)
cv.imshow("Binary Threshold (otsu)", output_otsuthresh)
cv.imwrite('output/rice_otsuthresh.png', output_otsuthresh)


# local adaptive thresholding - computes local threshold based on given window size
output_adapthresh = cv.adaptiveThreshold (input_rice, 255.0,
		cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -20.0)
cv.imshow("Adaptive Thresholding", output_adapthresh)
cv.imwrite('output/rice_adapthresh.png', output_adapthresh)


# morphologial operations - cleaning up binary images
kernel = np.ones((5,5),np.uint8)
# erosion
output_erosion = cv.erode(output_adapthresh, kernel)
cv.imshow("Morphological Erosion", output_erosion)
cv.imwrite('output/rice_erosion.png', output_erosion)
# dilation
output_dilation = cv.dilate(output_adapthresh, kernel)
cv.imshow("Morphological Dilation", output_dilation)
cv.imwrite('output/rice_dilation.png', output_dilation)
# opening - erosion followed by dilation
output_opening = cv.dilate(output_erosion, kernel)
cv.imshow("Morphological Opening", output_opening)
cv.imwrite('output/rice_opening.png', output_opening)
# closing - dilation followed by erosion
output_closing = cv.erode(output_dilation, kernel)
cv.imshow("Morphological Closing", output_closing)
cv.imwrite('output/rice_closing.png', output_closing)


# connected components - counts and marks number of distinct foreground objects
# apply connected components on clean binary image
# (in this case - output of morphological opening operation)
label_image = output_opening.copy()
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
cv.imwrite('output/rice_components.png', label_image)


# Contours - Computes polygonal contour boundary of foreground objects
# apply connected components on clean binary image
_, contours, _ = cv.findContours(output_opening, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
output_contour = cv.cvtColor(output_opening, cv.COLOR_GRAY2BGR)
cv.drawContours(output_contour, contours, -1, (0,0,255), 2)
print("Number of detected contours", len(contours))
cv.imshow("Contours", output_contour)
cv.imwrite('output/rice_contours.png', output_contour)


# region growing - segments image starting from seed points iteratively
# apply region growing on lung CT image
output_regiongrow = input_lungCT.copy()
cv.floodFill(output_regiongrow, None, (265, 225), (0, 64, 192), (4, 4, 4), (5, 5, 5))
cv.imshow("Region Growing", output_regiongrow)
cv.imwrite('output/lungCT_regiongrow.png', output_contour)


# wait for key press
cv.waitKey(0)
cv.destroyAllWindows()
