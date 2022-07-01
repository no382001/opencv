import cv2 as cv
import numpy as np
#22.morphological outline
img =  cv.imread('1.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
thresh_inv = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
blur = cv.GaussianBlur(thresh_inv,(1,1),0)
thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

mask = np.ones(img.shape[:2], dtype="uint8") * 255
for c in contours:
    # get the bounding rect
    x, y, w, h = cv.boundingRect(c)
    if w*h>1000:
        cv.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)

res_final = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))

gray=cv.cvtColor(res_final,cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(gray,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:]
idx =0 
for cnt in contours:
    idx += 1
    x,y,w,h = cv.boundingRect(cnt)
    roi=res_final[y:y+h,x:x+w]
    cv.imwrite(str(idx) + '.jpg', roi)
    #cv.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
cv.imshow('img',res_final)
cv.waitKey(0)