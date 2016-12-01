################
# lab4.py, a script to match feature detectors in images in openCV
# -Wesley Chavez
# 11/8/16
# Portland State University
# ECE 510: Embedded Vision
# Lab 4
################

import numpy as np
import cv2

# I adapted this function from 
# https://www.codementor.io/tips/5193438072/module-object-has-no-attribute-drawmatches-opencv-python
# and modified it to handle color images, since drawMatches isn't available with OpenCV 2.4

def drawMatches(img1, kp1, img2, kp2, matches, deleteme):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = img1

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (0, 0, 255), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (0, 0, 255), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)
    if (deleteme == 1):
        cv2.imwrite('bear_test1_surf_bf.jpg',out)
    if (deleteme == 2):
        cv2.imwrite('bear_test2_surf_bf.jpg',out)
    if (deleteme == 3):
        cv2.imwrite('shoe_test1_surf_bf.jpg',out)
    if (deleteme == 4):
        cv2.imwrite('shoe_test2_surf_bf.jpg',out)
    if (deleteme == 5):
        cv2.imwrite('bear_test1_orb_bf.jpg',out)
    if (deleteme == 6):
        cv2.imwrite('bear_test2_orb_bf.jpg',out)
    if (deleteme == 7):
        cv2.imwrite('shoe_test1_orb_bf.jpg',out)
    if (deleteme == 8):
        cv2.imwrite('shoe_test2_orb_bf.jpg',out)
    if (deleteme == 9):
        cv2.imwrite('bear_test1_surf_flann.jpg',out)
    if (deleteme == 10):
        cv2.imwrite('bear_test2_surf_flann.jpg',out)
    if (deleteme == 11):
        cv2.imwrite('shoe_test1_surf_flann.jpg',out)
    if (deleteme == 12):
        cv2.imwrite('shoe_test2_surf_flann.jpg',out)
    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Read training and testing images
bear = cv2.imread('lab4_images/bear.jpg')
bear_test1 = cv2.imread('lab4_images/bear_test1.jpg')
bear_test2 = cv2.imread('lab4_images/bear_test2.jpg')
shoe = cv2.imread('lab4_images/shoe.jpg')
shoe_test1 = cv2.imread('lab4_images/shoe_test1.jpg')
shoe_test2 = cv2.imread('lab4_images/shoe_test2.jpg')

# SURF detector, outputs keypoints and descriptors
surf = cv2.SURF(10000)
kp_bear_surf, des_bear_surf = surf.detectAndCompute(bear,None)
kp_bear_test1_surf, des_bear_test1_surf = surf.detectAndCompute(bear_test1,None)
kp_bear_test2_surf, des_bear_test2_surf = surf.detectAndCompute(bear_test2,None)
kp_shoe_surf, des_shoe_surf = surf.detectAndCompute(shoe,None)
kp_shoe_test1_surf, des_shoe_test1_surf = surf.detectAndCompute(shoe_test1,None)
kp_shoe_test2_surf, des_shoe_test2_surf = surf.detectAndCompute(shoe_test2,None)

# ORB detector, outputs keypoints and descriptors
orb = cv2.ORB()
kp_bear_orb, des_bear_orb = orb.detectAndCompute(bear,None)
kp_bear_test1_orb, des_bear_test1_orb = orb.detectAndCompute(bear_test1,None)
kp_bear_test2_orb, des_bear_test2_orb = orb.detectAndCompute(bear_test2,None)
kp_shoe_orb, des_shoe_orb = orb.detectAndCompute(shoe,None)
kp_shoe_test1_orb, des_shoe_test1_orb = orb.detectAndCompute(shoe_test1,None)
kp_shoe_test2_orb, des_shoe_test2_orb = orb.detectAndCompute(shoe_test2,None)

# Brute force matcher for SURF and ORB detectors
bf = cv2.BFMatcher()
matches_bear_test1_surf_bf = bf.match(des_bear_surf,des_bear_test1_surf)
matches_bear_test2_surf_bf = bf.match(des_bear_surf,des_bear_test2_surf)
# Sort by distance
matches_bear_test1_surf_bf = sorted(matches_bear_test1_surf_bf, key = lambda x:x.distance)
matches_bear_test2_surf_bf = sorted(matches_bear_test2_surf_bf, key = lambda x:x.distance)
# Draw first ten matches
result_bear_test1_surf_bf = drawMatches(bear,kp_bear_surf,bear_test1,kp_bear_test1_surf,matches_bear_test1_surf_bf[:10],1)
result_bear_test2_surf_bf = drawMatches(bear,kp_bear_surf,bear_test2,kp_bear_test2_surf,matches_bear_test2_surf_bf[:10],2)
matches_shoe_test1_surf_bf = bf.match(des_shoe_surf,des_shoe_test1_surf)
matches_shoe_test2_surf_bf = bf.match(des_shoe_surf,des_shoe_test2_surf)
# Sort by distance
matches_shoe_test1_surf_bf = sorted(matches_shoe_test1_surf_bf, key = lambda x:x.distance)
matches_shoe_test2_surf_bf = sorted(matches_shoe_test2_surf_bf, key = lambda x:x.distance)
# Draw first ten matches
result_shoe_test1_surf_bf = drawMatches(shoe,kp_shoe_surf,shoe_test1,kp_shoe_test1_surf,matches_shoe_test1_surf_bf[:10],3)
result_shoe_test2_surf_bf = drawMatches(shoe,kp_shoe_surf,shoe_test2,kp_shoe_test2_surf,matches_shoe_test2_surf_bf[:10],4)

matches_bear_test1_orb_bf = bf.match(des_bear_orb,des_bear_test1_orb)
matches_bear_test2_orb_bf = bf.match(des_bear_orb,des_bear_test2_orb)
matches_bear_test1_orb_bf = sorted(matches_bear_test1_orb_bf, key = lambda x:x.distance)
matches_bear_test2_orb_bf = sorted(matches_bear_test2_orb_bf, key = lambda x:x.distance)
result_bear_test1_orb_bf = drawMatches(bear,kp_bear_orb,bear_test1,kp_bear_test1_orb,matches_bear_test1_orb_bf[:10],5)
result_bear_test2_orb_bf = drawMatches(bear,kp_bear_orb,bear_test2,kp_bear_test2_orb,matches_bear_test2_orb_bf[:10],6)
matches_shoe_test1_orb_bf = bf.match(des_shoe_orb,des_shoe_test1_orb)
matches_shoe_test2_orb_bf = bf.match(des_shoe_orb,des_shoe_test2_orb)
matches_shoe_test1_orb_bf = sorted(matches_shoe_test1_orb_bf, key = lambda x:x.distance)
matches_shoe_test2_orb_bf = sorted(matches_shoe_test2_orb_bf, key = lambda x:x.distance)
result_shoe_test1_orb_bf = drawMatches(shoe,kp_shoe_orb,shoe_test1,kp_shoe_test1_orb,matches_shoe_test1_orb_bf[:10],7)
result_shoe_test2_orb_bf = drawMatches(shoe,kp_shoe_orb,shoe_test2,kp_shoe_test2_orb,matches_shoe_test2_orb_bf[:10],8)

FLANN_INDEX_KDTREE = 0
index_params_surf = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann_surf = cv2.FlannBasedMatcher(index_params_surf,search_params)
# Flann matcher for SURF detectors.
# I couldn't produce flann matches for orb descriptors because 
# "NameError: name 'FLANN_INDEX_LSH' is not defined"
# I couldn't hunt down the solution.
matches_bear_test1_surf_flann = flann_surf.match(des_bear_surf,des_bear_test1_surf)
matches_bear_test2_surf_flann = flann_surf.match(des_bear_surf,des_bear_test2_surf)
matches_bear_test1_surf_flann = sorted(matches_bear_test1_surf_flann, key = lambda x:x.distance)
matches_bear_test2_surf_flann = sorted(matches_bear_test2_surf_flann, key = lambda x:x.distance)
result_bear_test1_surf_flann = drawMatches(bear,kp_bear_surf,bear_test1,kp_bear_test1_surf,matches_bear_test1_surf_flann[:10],9)
result_bear_test2_surf_flann = drawMatches(bear,kp_bear_surf,bear_test2,kp_bear_test2_surf,matches_bear_test2_surf_flann[:10],10)
matches_shoe_test1_surf_flann = flann_surf.match(des_shoe_surf,des_shoe_test1_surf)
matches_shoe_test2_surf_flann = flann_surf.match(des_shoe_surf,des_shoe_test2_surf)
matches_shoe_test1_surf_flann = sorted(matches_shoe_test1_surf_flann, key = lambda x:x.distance)
matches_shoe_test2_surf_flann = sorted(matches_shoe_test2_surf_flann, key = lambda x:x.distance)
result_shoe_test1_surf_flann = drawMatches(shoe,kp_shoe_surf,shoe_test1,kp_shoe_test1_surf,matches_shoe_test1_surf_flann[:10],11)
result_shoe_test2_surf_flann = drawMatches(shoe,kp_shoe_surf,shoe_test2,kp_shoe_test2_surf,matches_shoe_test2_surf_flann[:10],12)

