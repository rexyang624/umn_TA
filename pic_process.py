#!/usr/bin/python

import numpy as np

def padding_image(image, width, height):
    (h,w) = image.shape[:2]
    padding = np.zeros((height,width)).astype(int)
    if not h%2 == 0:
        h += 1
        image = np.vstack((image,np.zeros(w))).astype(int)
    if not w%2 == 0:
        w += 1
        image = np.column_stack((image,np.zeros(h))).astype(int)
    if h > width or w > height:
        return padding
    padding[int(height/2)-int(h/2):int(height/2)+int(h/2),int(width/2)-int(w/2):int(width/2)+int(w/2)] = padding[int(height/2)-int(h/2):int(height/2)+int(h/2),int(width/2)-int(w/2):int(width/2)+int(w/2)]+image
    return padding.astype('uint8')
