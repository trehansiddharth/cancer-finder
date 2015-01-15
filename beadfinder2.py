# BeadFinder 2
import cv2
import numpy as np
import sys

def find_circles(img, small_r, big_r, circles = None):

    img = img.copy()

    if circles != None:
        print 'fill'
        fill_circles(img, circles)

    cv2.imwrite('filled.jpg', img)

    blur = odd((img.shape[0] + img.shape[1]) * 0.02)
    img = cv2.GaussianBlur(img, (blur, blur), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('blurred.jpg', img)

    thresh = img.mean()

    img = cv2.adaptiveThreshold(img,
                                255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV,
                                odd((img.shape[0] + img.shape[1]) * 0.05),
                                thresh / 4)



    cv2.imwrite('thresh.jpg', img)

    circles = cv2.HoughCircles(img,
                               cv2.cv.CV_HOUGH_GRADIENT,
                               3,
                               2 * small_r,
                               minRadius = small_r,
                               maxRadius = big_r)
    
    return circles

def odd(num):
    num = int(num)
    print (num / 2) * 2 + 1
    return (num / 2) * 2 + 1

def draw_circles(output, circles):
    for (x, y, r) in circles[0]:
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)

def fill_circles(output, circles, color = (0,0,0)):
    for (x, y, r) in circles[0]:
        cv2.circle(output, (x, y), r, color, -1)

def main():

    filename = sys.argv[1]
    print "Filename:", filename


    img = cv2.imread(filename)

    circles = find_circles(img, 30, 80)

    filled_img = img.copy()
    fill_circles(filled_img, circles)

    cv2.imwrite('filled.jpg', filled_img)

    circles = find_circles(img, 30, 80, circles)

    out = img.copy()
    draw_circles(out, circles)

    cv2.imwrite('circles-%s.jpg' % (filename), out)

main()