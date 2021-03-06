#!/usr/bin/python

import sys
import os
import cv2
import imutils
import numpy as np
from pic_process import padding_image
from matplotlib import pyplot as plt

img = cv2.imread(str(sys.argv[1]),0)

#Locate the ID area
id_area = img[240:290,830:1100]
blur = cv2.bilateralFilter(id_area,0,35,70)
id_process = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY_INV)[1]
contours = cv2.findContours(id_process.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]
#plt.imshow(id_process,'gray')
#plt.show()
id_digits = []
for contour in contours:
    (x ,y, w, h) = cv2.boundingRect(contour)
    #Find the viable number digits based on the size of contour
    if (w <=5  and h >= 16) or (w >= 15 and h >= 15):
       # Tried dividing connected digits by their width...
       # Further investigation needed...
       #
       # if float(w) / float(h) > 0.9:
       #     half_width = int(w / 2)
       #     id_digits.append((x, y, half_width, h))
       #     id_digits.append((x+half_width, y, half_width, h))
       # else:
        id_digits.append((x, y, w, h))
if not len(id_digits) == 7:
    error_msg = 'Unable to locate ID from '+repr(sys.argv[1])
    sys.exit(error_msg)
id_digits.sort(key = lambda x:x[0])
count = 1
if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])
for digit in id_digits:
    x, y, w, h = digit
    #resize_img = cv2.resize(resize_to_fit(id_process[y:y + h,x:x + w],64,64),(28,28))
    #plt.show()
    id_digit_padded = padding_image(id_process[y:y + h,x:x + w],50,50)
    cv2.imwrite(str(sys.argv[2])+str(count)+".jpg", cv2.resize(id_digit_padded,(28,28)))
    count = count + 1
    #plt.imshow(id_process[y:y + h,x:x + w],'gray')
    #plt.show()
