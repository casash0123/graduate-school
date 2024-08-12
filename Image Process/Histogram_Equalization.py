import cv2
import numpy as np
import matplotlib.pyplot as plt

image_origin_bgr = cv2.imread('../Data/Lenna.png')
image_origin_rgb = cv2.cvtColor(image_origin_bgr, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(image_origin_rgb)
image_gray = b

while True:
    color = input("Enter a color (r, g, b): ")

    if color.lower() in ['r', 'g', 'b']:
        if color.lower() == 'r':            image_gray = r
        elif color.lower() == 'g':          image_gray = g
        break

image_gray_equal = cv2.equalizeHist(image_gray)
image_hsv = cv2.cvtColor(image_origin_bgr, cv2.COLOR_BGR2HSV)

image_hsv_equal = image_hsv.copy()
image_hsv_equal[..., 2] = cv2.equalizeHist(image_hsv[..., 2])
image_hsv_equal_bgr = cv2.cvtColor(image_hsv_equal, cv2.COLOR_HSV2BGR)

plt.subplot(2, 5, 1)
plt.imshow(image_origin_rgb, cmap=None)
plt.title('original image')

plt.subplot(2, 5, 2)
plt.imshow(image_gray, cmap='gray')
plt.title('selected gray')

plt.subplot(2, 5, 3)
plt.imshow(image_gray_equal, cmap='gray')
plt.title('equalized gray')

plt.subplot(2, 5, 4)
plt.imshow(image_hsv, cmap=None)
plt.title('HSV image')


plt.subplot(2, 5, 5)
plt.imshow(image_hsv_equal_bgr, cmap=None)
plt.title('equalized-V HSV')

plt.subplot(2, 5, 7)
hist, bins = np.histogram(image_gray, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')

plt.subplot(2, 5, 8)
hist, bins = np.histogram(image_gray_equal, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('equalized value')

plt.subplot(2, 5, 9)
hist, bins = np.histogram(image_hsv, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('HSV value')

plt.subplot(2, 5, 10)
hist, bins = np.histogram(image_hsv_equal_bgr, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('equalized-V value')

plt.show()
