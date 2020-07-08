import cv2
import bpy
import bpycv
import random
import os
#import numpy as np

output_dir = "../Output_sec"

def mask_from_depth(img):
   # img = np.array(img)
    h, w = img.shape
    for r in range(h):
        for c in range(w):
            if(img[r,c] < 100):
                img[r,c] = 255
            else:
                img[r,c] = 0
                
    mask = img
    return mask


result = bpycv.render_data()

depth = result["depth"] / result["depth"].max() * 255
mask = mask_from_depth(depth)
os.system("mkdir "+output_dir)
cv2.imwrite(output_dir+"/mask.png",mask)
