import numpy as np
import cv2

def composition(img1,img2): #1: Background  2:iris
    pts1 = np.float32([[47, 191], [307, 219], [47, 440], [308, 407]])
    #pts1 = np.float32([[360, 94], [738, 46], [363, 291], [735, 306]]) back2
    #pts1 = np.float32([[360, 240], [498, 235], [358, 333], [498, 330]]) back3
    pts2 = np.float32([[0, 0], [300, 0], [0, 250], [300, 250]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    M_inv = np.linalg.inv(M)

    dst = cv2.warpPerspective(img1, M, (300, 250))
    iris2 = cv2.resize(img2, (300, 250))
    dst2 = cv2.warpPerspective(iris2, M_inv, (800, 560))

    copyImg = dst2.copy()
    h, w = copyImg.shape[:2]

    mask = np.ones([h+2,w+2,1],np.uint8)
    mask[h+1:w+1,h+1:w+1] = 0
    cv2.floodFill(copyImg, mask, (30, 30), (255, 255, 255), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    img2gray = cv2.cvtColor(copyImg, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 0.5, 255, cv2.THRESH_BINARY_INV)
    rows, cols, channels = copyImg.shape
    y = 0
    x = 0
    roi = img1[y:0 + rows, x:0 + cols]
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask)

    mask_inv = cv2.bitwise_not(mask)
    img2_fg = cv2.bitwise_and(copyImg, copyImg, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    img1[y:y + rows, x:x + cols] = dst
    return img1


img1 = cv2.imread('Background1.bmp')
img2 = cv2.imread('iris.jpg')
new_img = composition(img1,img2)
cv2.imshow('Background1.jpg', new_img)
cv2.waitKey(0)
cv2.imwrite( 'Background1.jpg', new_img)
cv2.destroyAllWindows()