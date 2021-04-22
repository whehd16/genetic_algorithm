import numpy as np
import random, pickle
from tqdm import tqdm
import cv2, time, argparse

pts = []
img_size = [384, 384]

img = np.zeros((384,384,3), np.uint8)

max_shapes = 100
point = [np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0])]
pts.append(point)

# n_shapes = np.random.randint(0, max_shapes)
n_shapes=3

for i in range(n_shapes):
    new_point = [point[0] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[1] / 4)), point[1] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[0] / 4))]
    pts.append(new_point)

pts 	 = np.array(pts)
pts 	 = pts.reshape((-1, 1, 2))
print(pts)
overlay = img.copy()
cv2.imwrite("./0_img.jpg", overlay)

# points = np.array([[10,10],[170,10],[200,230],[70,70],[40,150]], np.int32)
overlay = cv2.fillPoly(overlay, [pts], [255, 0, 0], 8)

# [np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)]
# cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

cv2.imwrite("./1_img.jpg", overlay)
