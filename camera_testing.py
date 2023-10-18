# from cv2 import *
# Port_#0002.Hub_#0003
import cv2

cv2.namedWindow("Window")
vc = cv2.VideoCapture(0)

if vc.isOpened():   # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("Window", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("Window")
