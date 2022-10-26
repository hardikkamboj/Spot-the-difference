import cv2 
import numpy as np 
from helper_functions import align_images, get_top_k_differences

im1 = cv2.imread('diff1.png') # this image will be transformed         
im2 = cv2.imread('diff2.png')  # this image will be used as a reference 

input_image = np.concatenate((im1, im2), axis=1)

if im1.shape != im2.shape:
    im1 = cv2.resize(im1, im2.shape[:2][::-1])

im1_aligned = align_images(im1, im2)

result = get_top_k_differences(im1_aligned, im2)

cv2.imshow("Aligned Image", input_image)
cv2.imshow("Result", result)
cv2.waitKey(0)