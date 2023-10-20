import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np

cv2.namedWindow("Press space to capture")
vc = cv2.VideoCapture(0)

if vc.isOpened():   # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

# print(grayscale_image[0][0][:])

while rval:
    cv2.imshow("Press space to capture", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 32:   # exit on ESC
        break

vc.release()
cv2.destroyWindow("Press space to capture")

grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape(480, 640)

for i in range(len(grayscale_image)):
    for j in range(len(grayscale_image[0])):
        grayscale_image[i][j] = 255 - grayscale_image[i][j]

resized_gray_image = cv2.resize(grayscale_image, (28, 28), cv2.INTER_LINEAR)

plt.imshow(resized_gray_image)
plt.show()

resized_gray_image = resized_gray_image.reshape(1, 28, 28, 1)

ld_model = load_model('data/cnn.keras')
prediction = ld_model.predict(resized_gray_image)

print(prediction)
print(np.argmax(prediction))
