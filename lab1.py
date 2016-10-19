################
# lab1.py, a script to manipulate images with openCV
# -Wesley Chavez
# 10/4/16
# Portland State University
# ECE 510: Embedded Vision
# Lab 1
################

# Import libraries
import numpy as np
import cv2

# Read image file and determine dimensions
imgfile = 'upsidedown.jpg'
img = cv2.imread(imgfile)
y=img.shape[0]
x=img.shape[1]

# Display the original image
cv2.imshow('image', img)
cv2.waitKey(0)

# Downsample by half in each dimension and save to file.  If no dstsize is 
# specified, pyrDown automatically downsamples by half in each dimension.
# pyrDown actually performs a Gaussian blur with a 5x5 kernel before getting
# rid of every other column and row. No artifacts are visible.
imgDS2 = cv2.pyrDown(img,dstsize = (x/2,y/2))
cv2.imshow('image', imgDS2)
cv2.waitKey(0)
cv2.imwrite('imageDownsampledBy2.jpg', imgDS2)

# Downsample again by half in each dimension and save to file.
imgDS4 = cv2.pyrDown(imgDS2,dstsize = (x/4,y/4))
cv2.imshow('image', imgDS4)
cv2.waitKey(0)
cv2.imwrite('imageDownsampledBy4.jpg', imgDS4)

# Upsample the downsampled image (imgDS4) by two in each dimension and save
# to file.
imgUS2 = cv2.pyrUp(imgDS4,dstsize = (x/2,y/2))
cv2.imshow('image', imgUS2)
cv2.waitKey(0)
cv2.imwrite('imageUpsampledBy2.jpg', imgUS2)

# Upsample again by two in each dimension and save to file. Because of the
# blurring of pyrDown and the loss of information when downsampling an
# image, upsampling to the original size results in a degradation in the
# form of blurriness.
imgUS4 = cv2.pyrUp(imgUS2,dstsize = (x,y))
cv2.imshow('image', imgUS4)
cv2.waitKey(0)
cv2.imwrite('imageUpsampledBy4.jpg', imgUS4)

# Create black (zeros) 200x200 pixel BGR image (multidimensional np array)
imgbg = np.zeros((200,200,3), np.uint8)

# Draw a circle on the image and display.  imgbg is the designated image, 
# (100,100) is the center point of the circle, 20 is the pixel radius, 
# (255,0,0) is the circle color (BGR, so blue circle), and -1 is the 
# thickness; any negative number in this argument results in a filled circle
cv2.circle(imgbg,(100,100),20,(255,0,0),-1)
cv2.imshow('image', imgbg)
cv2.waitKey(0)

# Similar to the .at function in C++, we can specify indices or a range of
# indices with which to specify a value.  60-101 in the first dimension 
# (loops across rows), 30:101 in the second dimension (loops across
# columns), and 1 in the third dimension (BGR, so green channel). 60-101
# actually indicates rows 60-100 (0-indexed), which is why I like MATLAB
# better for this sort of thing, and it's 1-indexed, so in MATLAB I would
# say imgbg[60:100,30:100,2] = 255, which is more intuitive to me.  All the
# pixels in this range are at maximum intensity (255 for uint8)  
imgbg[60:101,30:101,1] = 255
cv2.imshow('image', imgbg)
cv2.waitKey(0)
cv2.imwrite('BlueCircleGreenRectangle.jpg', imgbg)

# Since I created a circle on the blue channel and a rectangle on the green # channel, pixels where these shapes overlap have color values (255,255,0), # which is cyan because RGB/BGR is additive.
