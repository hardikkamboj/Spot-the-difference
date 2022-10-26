import cv2 
import numpy as np 

def isBorderCont(cnt,img):
    M = cv2.moments(cnt)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    h,w, _ = img.shape
    if x/w > 0.98:
        return True 
    elif x/w < 0.02:
        return True 
    elif y/h>0.98:
        return True 
    elif y/h < 0.02:
        return True 
    else:
        return False 


def align_images(input_image, reference_image):
    """Given two images where first image is the input image and second image is the reference image, it aligns the input image to the reference image and returns the output image.

    Args:
        input_image (numpy array): first image which is transformed 
        reference_image (numpy array): the reference used to align the input image

    Returns:
        numpy array: the transformed form of the input image
    """
    gray1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # craete an object of orb create 
    orb = cv2.ORB_create(5000)  

    kp1, des1 = orb.detectAndCompute(gray1, None)  #kp1 --> list of keypoints
    kp2, des2 = orb.detectAndCompute(gray2, None)

    ## this is to view the results of the keypoints detected 
    # result1 = cv2.drawKeypoints(np.copy(input_image), kp1, None)
    # result2 = cv2.drawKeypoints(np.copy(reference_image), kp2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    # Match descriptors.
    matches = matcher.match(des1, des2, None) 

    matches = sorted(matches, key = lambda x:x.distance)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography
    height, width, channels = reference_image.shape
    img_aligned = cv2.warpPerspective(input_image, h, (width, height))  #Applies a perspective transformation to an image (img_1)

    return img_aligned


def get_top_k_differences(image_1, image_2, k=7):
    """Given two images, it finds the difference and returns the result image containing the top k most biggest differences highlighted

    Args:
        image_1 (np.ndarray): first image 
        image_2 (np.ndarray): seconda image 
        k (int, optional): the number of differences wanted. Defaults to 7.

    Returns:
        np.ndarray: Resultant image, two images concantenated side by side  
    """
    sub = cv2.subtract(image_1, image_2)
    b,g,r = cv2.split(sub)
    img_max = np.max((b,g,r), axis=0)

    ret, thrsh = cv2.threshold(img_max, 127, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thrsh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_conts = sorted(contours, key = cv2.contourArea, reverse=True)

    sorted_conts_k = sorted_conts[:k]

    removed_border_cnts = []
    for cnt in sorted_conts_k:
        if not isBorderCont(cnt, sub):
            removed_border_cnts.append(cnt)

    result_1 = image_1.copy()
    result_2 = image_2.copy()

    for c in removed_border_cnts:       
        poly = cv2.approxPolyDP(c, 3, True)
        center, radius = cv2.minEnclosingCircle(poly)
        # cv2.circle accepts circle as int 
        center = (int(center[0]), int(center[1])) 
        cv2.circle(result_1, center, int(radius), (0,0,255), 4)
        cv2.circle(result_2, center, int(radius), (0,0,255), 4)

    final_result = np.concatenate((result_1, result_2), axis=1)
    return final_result
