################
# lab2.py, a script to manipulate images with openCV
# -Wesley Chavez
# 10/13/16
# Portland State University
# ECE 510: Embedded Vision
# Lab 2
################

# Task #1
# Filtering an image I of size Ncols x Nrows x 3 with a kernel K of size 
# k x k will take (2*k*k-1)*Nrows*Ncols*3 total operations. The reason for 
# 2*k*k-1 is because I'm assuming an accumulator adds the result of the last
# multiply operation to itself.  For example, a 2x2 kernel would need to
# compute (K_00*I_00 + K_01*I_01 + K_10*I_10 + K_11*I_11).  Adding up the
# operations in this equation is 7, 2*k*k-1.  Running a 1D filter
# horizontally then vertically would take (2*k-1)*Nrows*Ncols*3*2 total
# operations, the last (*2) because of the two orientations with which to
# convolve the kernel over the image.  This way seems more computationally
# efficient than a full 2D convolution. 


# Import libraries
import numpy as np
import cv2

# Read image as monochrome
img = cv2.imread('upsidedown.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# Take the negative
negimg = 255 - img
# Define a kernel size with which to dilate/erode
kernel = np.ones((5,5),np.uint8)
# Erode the negative
eroded = cv2.erode(negimg,kernel,iterations = 1)
# Then take the negative
posimg = 255 - eroded
# Dilate the original image
dilated = cv2.dilate(img,kernel,iterations = 1)
# If erosion and dilation are symmetrical, posimg and dilated
# will be equal
print (np.array_equal(posimg,dilated))

# For comprehensivity's sake, dilate the negative image
dilated = cv2.dilate(negimg,kernel,iterations = 1)
# Take the negative
posimg = 255 - dilated
# Erode the original image
eroded = cv2.erode(img,kernel,iterations = 1)
# If erosion and dilation are symmetrical, eroded and posimg
# will be equal
print (np.array_equal(eroded,posimg))

# Dilation minus erosion on the same image gives us the
# morphological gradient
dilated = cv2.dilate(img,kernel,iterations = 1)
eroded = cv2.erode(img,kernel,iterations = 1)
morphGrad = dilated - eroded

# Morphological gradient on the negative image
dilatedNeg = cv2.dilate(negimg,kernel,iterations = 1)
erodedNeg = cv2.erode(negimg,kernel,iterations = 1)
morphGradNeg = dilatedNeg - erodedNeg

# If these are equal, morphological gradient gives us the same
# edges on the positive and negative images
print (np.array_equal(morphGrad,morphGradNeg))

# Dilating the eroded image opens it
opened = cv2.dilate(eroded,kernel,iterations = 1)
# Eroding the dilated negative of the image closes it
closedNeg = cv2.erode(dilatedNeg,kernel,iterations = 1)
# Take the negative again...
negClosedNeg = 255 - closedNeg

# If the operations are symmetrical, will print TRUE
print (np.array_equal(opened,negClosedNeg))

# Picture without coffee cup, grayscale
nocoffee = cv2.imread('nocoffee.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# Picture with coffee cup, grayscale
coffee = cv2.imread('coffee.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Need to cast to signed 16 bit ints to subtract
nocoffee16 = nocoffee.astype(np.int16)
coffee16 = coffee.astype(np.int16)

# Subtract and take the absolute value
diff16 = np.absolute(coffee16 - nocoffee16)
# Convert back to uint8; we know all the values are within 0-255
diff = diff16.astype(np.uint8)

cv2.imshow('image0', diff) 
cv2.waitKey(0) 
cv2.imwrite('Differenced.png',diff)

# Found a good threshold for the differenced image empirically
binaryThresh = 100
diff[diff>binaryThresh] = 255
diff[diff<=binaryThresh] = 0

cv2.imshow('image0', diff) 
cv2.waitKey(0) 
cv2.imwrite('BinaryThresholded.png',diff)

# Closed the image A LOT (50 iterations) to fill in parts of the mask
# It is one clean shape because of this, but isn't PERFECT
dilated = cv2.dilate(diff,kernel,iterations= 50)
closed = cv2.erode(dilated,kernel,iterations = 50)

cv2.imshow('image0', closed) 
cv2.waitKey(0) 
cv2.imwrite('Closed.png',closed)

# Loop through the image (columns loop the quickest)
# and use the first pixel that's 255 as an anchor
# point for a flood fill
# It worked because there were no scattered ON pixels around the mask
ex = 0
for i in np.arange(closed.shape[0]):
    # ex is so we can break out of both loops after finding the pixel
    # and flood-filling it
    if ex == 1:
        break
    for j in np.arange(closed.shape[1]):
        if closed[i,j] == 255:
            # Flood fill's mask has to be 2 pixels higher and wider than the
            # original image
            mask = np.zeros((closed.shape[0]+2, closed.shape[1]+2), np.uint8)
            # When we find the first pixel, flood-fill the shape to a value
            # of 100 and break the for loop (both for loops because ex = 1)
            cv2.floodFill(closed,mask,(i,j),100)
            ex = 1
            break

# If closing doesn't work to get a nice-looking mask, try deep learning to
# find the cup
cv2.imshow('image0', closed) 
cv2.waitKey(0) 
cv2.imwrite('FloodFilled.png', closed)

# Set the 100 pixels back to 255 for bitwise AND
closed[closed==100] = 255
# Bitwise and original with coffee cup and the mask
and0 = np.bitwise_and(coffee,closed)
# Complement of the mask (negative)
closed = 255 - closed
# Bitwise and original with NO coffee cup and the mask complement
and1 = np.bitwise_and(nocoffee,closed)
# Bitwise OR those two. Result is the coffee cup where we think the
# coffee cup is (the mask) inserted into the picture without the cup
composite = np.bitwise_or(and0,and1)


cv2.imwrite('Composite.png', composite)
cv2.imshow('image0', composite) 
cv2.waitKey(0) 
