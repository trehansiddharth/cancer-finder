# BeadFinder 2
import cv2
import numpy as np

def find_circles(img, small_r, big_r):
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img,
                               cv2.cv.CV_HOUGH_GRADIENT,
                               3,
                               2 * small_r,
                               minRadius = small_r,
                               maxRadius = big_r)
    
    return circles

def draw_circles(output, circles):
    for (x, y, r) in circles[0]:
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)



def main():
    img = cv2.imread("sample1.jpg")
    output = img.copy()

    circles = find_circles(img, 40, 60)

    draw_circles(output, circles)

    cv2.imwrite('circles.jpg', output)

main()