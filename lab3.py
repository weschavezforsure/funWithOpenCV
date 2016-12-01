################
# lab3.py, a script to find faces in images in openCV
# -Wesley Chavez
# 10/23/16
# Portland State University
# ECE 510: Embedded Vision
# Lab 3
################

import numpy as np
import cv2

group = cv2.imread('group.jpg')
    # To test whether back-projection works with three dimensions, let's take a
    # histogram of a face within the target image (not from face_template.jpg)
face = group[208:233,178:196,:]
    # Implements colorReduce to four colors per channel
group = (group&192) + 32
face = (face&192) + 32
    # Calculate a histogram in all three channels of the histogrammed image
facehist = cv2.calcHist([face], [0,1,2], None, [256,256,256], [0,255,0,255,0,255])
    # Back-project group image with the template histogram
dst = cv2.calcBackProject([group],[0,1,2],facehist,[0,255,0,255,0,255],1)
    
    # Maximum of back-projection is 0. (Not good)
    # I tried tinkering with the code to calculate back-projection in BGR space,
    # but it seems that there is a bug in the calculation of back-projection
    # for 3D histograms.
    # A few things to note: I extensively tested the colorReduce code, and made sure 
    # the histogram was producing reasonable results, read and re-read the 
    # documentation for calcHist and calcBackProject, and spent way too long trying to
    # get this part to work in accordance with the lab specs. However, in the process
    # of researching uses of these functions, I learned that HSV color space is more
    # conducive to back-projections in color images than BGR for many tasks... 
print np.amax(dst)

    # This code is adapted from http://docs.opencv.org/trunk/dc/df6/tutorial_py_histogr
    # am_backprojection.html
    # Read image and use one of the faces in the image for the histogram
group = cv2.imread('group.jpg')
face = group[208:233,178:196,:]
    # Color reduce
group = (group&192) + 32
face = (face&192) + 32
    # Convert  color-reduced images to HSV
grouphsv = cv2.cvtColor(group,cv2.COLOR_BGR2HSV)
facehsv = cv2.cvtColor(face,cv2.COLOR_BGR2HSV)
    # H ans S, but not V (a 2D histogram) is more widely used than BGR.
facehist = cv2.calcHist([facehsv], [0,1], None, [180,256], [0,180,0,256])
    # Normalize the histogram from 0 to 255    
cv2.normalize(facehist, facehist, 0, 255, cv2.NORM_MINMAX)
    # Calculate back-projection
dst = cv2.calcBackProject([grouphsv],[0,1],facehist,[0,180,0,256],1)
    # Normalize from 0 to 255
cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
    # And threshold at 250
ret,thresh1 = cv2.threshold(dst,250,255,cv2.THRESH_BINARY)
    # Display and write image
cv2.imshow('thresh1',thresh1)
cv2.waitKey(0)
cv2.imwrite('BackProjection_InSameImage.png',thresh1)

    # The face template image does much worse as a face copied from the actual
    # group picture, at its optimal threshold
group = cv2.imread('group.jpg')
face = cv2.imread('face_template.jpg')
    # Convert to HSV
grouphsv = cv2.cvtColor(group,cv2.COLOR_BGR2HSV)
facehsv = cv2.cvtColor(face,cv2.COLOR_BGR2HSV)
    # Calculate histogram, normalize, back-project, and normalize again
facehist = cv2.calcHist([facehsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(facehist, facehist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([grouphsv],[0,1],facehist,[0,180,0,256],1)
cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
    # Threshold the result
ret,thresh2 = cv2.threshold(dst,80,255,cv2.THRESH_BINARY)
cv2.imshow('thresh2',thresh2)
cv2.waitKey(0)
cv2.imwrite('BackProjection_WithFaceTemplate.png',thresh2)

    # Quick note, I would calculate true positive, false positive, true negative,
    # and false negative rates, and make an ROC, but that would require hand-annotating
    # every single pixel of the group image in order to compare with the results at
    # different thresholds, which would be too much to ask.


    # Template matching, load grayscale images
group = cv2.imread('group.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
face = cv2.imread('face_template.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # Match face template using square difference similarity
result = cv2.matchTemplate(group,face,cv2.TM_SQDIFF_NORMED)
    # Lower values means more similar, so take negative
result = 1.0-result
    # Find threshold empirically
ret,sqdiffnorm = cv2.threshold(result,.8,1.0,cv2.THRESH_BINARY)
    # Normalize and write image
cv2.normalize(sqdiffnorm, sqdiffnorm, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('sqdiffnorm',sqdiffnorm)
cv2.waitKey(0)
cv2.imwrite('SquareDiffNorm.png',sqdiffnorm)

    # Match face template using cross-correlation similarity, threshold, normalize,
    # and write image
result = cv2.matchTemplate(group,face,cv2.TM_CCORR_NORMED)
ret,ccorrnorm = cv2.threshold(result,.92,1.0,cv2.THRESH_BINARY)
cv2.normalize(ccorrnorm, ccorrnorm, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('ccorrnorm',ccorrnorm)
cv2.waitKey(0)
cv2.imwrite('CCorrNorm.png',ccorrnorm)

    # Match face template using cross-coefficient similarity, threshold, normalize,
    # and write image
result = cv2.matchTemplate(group,face,cv2.TM_CCOEFF_NORMED)
ret,ccoeffnorm = cv2.threshold(result,.4,1.0,cv2.THRESH_BINARY)
cv2.normalize(ccoeffnorm, ccoeffnorm, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('ccoeffnorm',ccoeffnorm)
cv2.waitKey(0)
cv2.imwrite('CCoeffNorm.png',ccoeffnorm)

    # Blurring the template may prove a better face detector
face = cv2.blur(face,(5,5))
    # Match blurred face template using square difference similarity, threshold,
    # normalize, and write image
result = cv2.matchTemplate(group,face,cv2.TM_SQDIFF_NORMED)
result = 1.0-result
ret,sqdiffnorm = cv2.threshold(result,.8,1.0,cv2.THRESH_BINARY)
cv2.normalize(sqdiffnorm, sqdiffnorm, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('sqdiffnorm_blurred',sqdiffnorm)
cv2.waitKey(0)
cv2.imwrite('SquareDiffNorm_Blurred.png',sqdiffnorm)

    # Match blurred face template using cross-correlation similarity, threshold,
    # normalize, and write image
result = cv2.matchTemplate(group,face,cv2.TM_CCORR_NORMED)
ret,ccorrnorm = cv2.threshold(result,.92,1.0,cv2.THRESH_BINARY)
cv2.normalize(ccorrnorm, ccorrnorm, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('ccorrnorm_blurred',ccorrnorm)
cv2.waitKey(0)
cv2.imwrite('CCorrNorm_Blurred.png',ccorrnorm)

    # Match blurred face template using cross-coefficient similarity, threshold,
    # normalize, and write image
result = cv2.matchTemplate(group,face,cv2.TM_CCOEFF_NORMED)
ret,ccoeffnorm = cv2.threshold(result,.4,1.0,cv2.THRESH_BINARY)
cv2.normalize(ccoeffnorm, ccoeffnorm, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('ccoeffnorm_blurred',ccoeffnorm)
cv2.waitKey(0)
cv2.imwrite('CCoeffNorm_Blurred.png',ccoeffnorm)

    # Let's see if a Canny edge detector will separate faces
group = cv2.imread('group.jpg')
    # Find optimal threshold empirically
edges = cv2.Canny(group,150,200)
    # The third and fourth parameters are the contour retrieval mode, and the contour
    # approximation method, respectively. 
contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # Make empty np array of same size as group to draw contours on
empty = np.zeros((457,900,3))
    # Draw the contours in blue (B,G,R).  The first -1 is the contour index, and -1
    # means all contours are drawn.  The second -1 is for line thickness, and -1
    # means fill in the contour interiors
cv2.drawContours(empty, contours, -1, (255,0,0),-1)
cv2.imshow('contours',empty)
cv2.waitKey(0)
cv2.imwrite('Contours.png',empty)
    # Some of the faces ended up as contours

