import numpy as np
import cv2 as cv
import imutils
from skimage.metrics import structural_similarity

##Reading the two images
img1 = cv.imread('image1.jpg')
img2 = cv.imread('image2.jpg')

# Resize the image
img1 = cv.resize( img1, [640, 640])
img2 = cv.resize( img2, [640, 640])

#Grayscaling
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

## Computing the strucutal similarity index
(score, diff) = structural_similarity(img1_gray, img2_gray, full=True)
print(f"iamge similarity score {score*100:.2f}")

## Converting the diff to int range [0,255]
diff = (diff * 255).astype("uint8")

## Find contours to get the regions that differ
thresh  = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#contours = contours[0] if len(contours) == 2 else contours[1]
contours = imutils.grab_contours(contours)

## Creating a mask of img's shape
mask = np.zeros(img1.shape, dtype='uint8')

## Create copy of second image to display the result on
result = img2.copy()

for c in contours:
    area = cv.contourArea(c)
    if area > 40:
        x,y,w,h = cv.boundingRect(c)
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2

        # Draw the circle on an image
        cv.circle(img1, (center_x, center_y), radius, (0,0 ,255), 2)
        cv.circle(img2, (center_x, center_y), radius, (0,0,255), 2)
        cv.drawContours(mask, [c], 0, (255,255,255), -1)
        cv.drawContours(result, [c], 0, (0,255,0), -1)

cv.imshow('Image 1', img1)
cv.imshow('Image 2', img2)
cv.imshow('difference of two images', diff)
cv.imshow('Mask', mask)
cv.imshow('filling the difference', result)
cv.waitKey()