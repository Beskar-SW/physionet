import cv2
import sys
import numpy as np

image_path = "./test.png"
scale_by = 0.5

img = cv2.imread(image_path)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#hsv_img = cv2.resize(hsv_img, None, fx=scale_by, fy=scale_by)

lower = np.array([0, 0, 0])
upper = np.array([180, 255, 150])
mask = cv2.inRange(hsv_img, lower, upper)

if img is None:
    sys.exit(1)

#cv2.imshow("Binary", mask)

cv2.waitKey(0)

cv2.imwrite("binary.png", mask)

cv2.destroyAllWindows()
