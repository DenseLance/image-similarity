import cv2
import sys
import numpy as np

np.set_printoptions(threshold = sys.maxsize)
img = cv2.imread("final.jpg")

##print(list(img))
np.save("img.npy", img)
