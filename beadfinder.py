import cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import *

def readImageGrayscale8Bit(imgPath):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    return gray

def outputImage(imgPath, img):
    cv2.imwrite(imgPath, img)

def polarize(img, (x, y)):
    raise NotImplementedError

def findBeads(inputPath, outputPath):
    img = readImageGrayscale8Bit(inputPath)

    cimg = cv2.Canny(img, 250, 350)

    #circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,20,
    #                        param1=50,param2=30,minRadius=0,maxRadius=0)

    #circles = np.uint16(np.around(circles))
    #for i in circles[0,:]:
    #    # draw the outer circle
    #    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #    # draw the center of the circle
    #    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    outputImage(outputPath, cimg)

def main():
    img = readImageGrayscale8Bit("sample1.jpg")
    template = readImageGrayscale8Bit("template1.jpg")
    w, h = template.shape
    print template.shape
    templateRadius = sqrt(w**2 + h**2)

    print templateRadius

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_SQDIFF_NORMED'] #, 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                #'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    results = []

    for meth in methods:
        method = eval(meth)

        # Apply template Matching
        result = cv2.matchTemplate(img,template,method)
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        threshold = 0.32
        loc = np.where(result <= threshold)
        
        for pt in zip(*loc[::-1]):
            x, y = pt
            addRectangle = True
            for (rx, ry) in results:
                distance = sqrt((rx - x)**2 + (ry - y)**2)
                if distance < (templateRadius / 3):
                    addRectangle = False
                    break
            if addRectangle:
                cv2.rectangle(img, pt, (int(x + 1.1 * w), int(y + 1.1 * h)), (0,0,255), 2)
                results.append(pt)
            
    outputImage("output.jpg", img)

findBeads("sample1.jpg", "output.jpg")
