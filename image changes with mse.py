import numpy as np
import cv2 as cv

##Reading the two images
img1 = cv.imread('image1.jpg')
img2 = cv.imread('image2.jpg')

# Resize the image
img1 = cv.resize( img1, [640, 640])
img2 = cv.resize( img2, [640, 640])

#Grayscaling
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

### Computing the MSE 
def mse(img1, img2):
    h, w = img1.shape
    diff = cv.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse, diff

error, diff = mse(img1, img2)
print(f"MSE Error bw two images", error)

cv.imshow('imageee1', img1)
cv.imshow('imageee2', img2)
cv.imshow("difference", diff)
cv.waitKey(0)