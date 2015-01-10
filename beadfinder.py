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

def integerDistance((x1, y1), (x2, y2)):
    return int(floor(sqrt((x1 - x2)**2 + (y1 - y2)**2)))

def polarizeAverage(img, p):
    w, h = img.shape
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    maxdistance = max(map(lambda q: integerDistance(p, q), corners))
    items = np.zeros(maxdistance)
    result = np.zeros(maxdistance)
    for index, value in np.ndenumerate(img):
        distance = integerDistance(index, p)
        items[distance] += 1
        result[distance] += value
    for index, value in np.ndenumerate(result):
        b = np.nan
        if items[index] > 0:
            b = int(float(result[index]) / float(items[index]))
        result[distance] = b
    return result

def polarizeVariance(img, p):
    w, h = img.shape
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    maxdistance = max(map(lambda q: integerDistance(p, q), corners))
    items = np.zeros(maxdistance)
    averages = polarizeAverage(img, p)
    variances = np.zeros(maxdistance)
    for index, value in np.ndenumerate(img):
        distance = integerDistance(index, p)
        error = (value - averages[distance])**2
        items[distance] += 1
        variances[distance] += error
    for index, value in np.ndenumerate(variances):
        b = np.nan
        if items[index] > 0:
            b = int(float(variances[index]) / float(items[index]))
        variances[distance] = b
    return variances

def applyKernel(array, x, kernel):
    size = len(kernel)
    if size % 2 == 0:
        raise IndexError
    start = x - (size - 1) / 2
    end = x + (size - 1) / 2
    result = 0
    for i in range(start, end + 1):
        if i >= 0 and i < array.shape[0]:
            result += kernel[i - start] * array[i]
    return result

def convolve(array, kernel):
    result = np.zeros(array.shape)
    for i in range(0, array.shape[0]):
        result[i] = applyKernel(array, i, kernel)
    return result

def derivative(array, kernel=[-0.5, 0.0, +0.5]):
    return convolve(array, kernel)

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
