# BeadFinder 2
import cv2
import numpy as np

def find_circles(img, small_r, big_r):
    blur = odd((img.shape[0] + img.shape[1]) * 0.02)
    img = cv2.GaussianBlur(img, (blur, blur), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('blur.jpg', img)
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



def main():
    img = cv2.imread("sample2.jpg")
    output = img.copy()

    circles = find_circles(img, 30, 80)

    draw_circles(output, circles)

    cv2.imwrite('circles.jpg', output)

main()