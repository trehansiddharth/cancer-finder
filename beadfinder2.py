# BeadFinder 2
import cv2
import numpy as np

def main():
    img = cv2.imread("sample2.jpg")
    #img = cv2.imread("template1.jpg")
    #img = cv2.imread("ideal-one-circle.jpg")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_g = 255 - img_g
    img_gray_sobel = cv2.Sobel(img_g, -1, 1, 1, ksize =3, scale = 20)
    #cv2.imwrite("sobel.jpg",img_gray_sobel)
    small_r = 60
    big_r  = small_r + 5
    pixels_wide = 2 * big_r + 1
    kernel = np.zeros((pixels_wide, pixels_wide))
    #print kernel.shape
    center = (big_r + 1, big_r + 1);
    count  = 0
    for r in range(kernel.shape[0]):
        for c in range(kernel.shape[1]):
            radius_sq = (center[0] - r) ** 2 + (center[1] - c) ** 2
            #print "[%d,%d] -> %d" % (r, c, radius_sq ** 0.5)
            if (radius_sq > small_r ** 2 and radius_sq <= big_r ** 2):
                #print 'match'
                kernel[r][c] = 1
                count += 1
    #cv2.imwrite("img_invert.jpg", img_g)
    #print 1. / count
    kernel = np.multiply(kernel, 1. / count)
    #print "Sum: %d" % (kernel.sum(),)
    filtered = cv2.filter2D(img_gray_sobel, -1, kernel) # -1 -> use default
    cv2.imwrite("output1.jpg", np.multiply(filtered, 1))
    #cv2.imshow('filter1', filtered)

main()