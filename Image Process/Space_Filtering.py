import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../Data/Lenna.png').astype(np.float32) / 255

noised = (image + 0.2 * np.random.rand(*image.shape).astype((np.float32)))
noised = noised.clip(0,1)


plt.subplot(2, 4, 1)
plt.imshow(noised[:,:,[2,1,0]])
plt.title("Noised")

plt.subplot(2, 4, 5)
plt.imshow(image[...,[2,1,0]])
plt.title('original image')


plt.subplot(2, 4, 2)
gauss_blur = cv2.GaussianBlur(noised, (7,7), 0)
gauss_blur = gauss_blur.clip(0, 1)
plt.imshow(gauss_blur[:,:,[2,1,0]])
plt.title("gauss")

plt.subplot(2, 4, 6)
diff = np.abs(gauss_blur - image)
plt.imshow(diff)
plt.title("gauss diff")


plt.subplot(2, 4, 3)
median_blur = cv2.medianBlur((noised*255).astype(np.uint8),7)
plt.imshow(median_blur[:,:,[2,1,0]])
plt.title("median")

plt.subplot(2, 4, 7)
median_blur = median_blur.clip(0, 1)
diff = np.abs(median_blur - image)
plt.imshow(diff)
plt.title("median - diff")


plt.subplot(2, 4, 4)
bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
bilat = bilat.clip(0, 1)
plt.imshow(bilat[:,:,[2,1,0]])
plt.title("bilat")

plt.subplot(2, 4, 8)
diff = np.abs(bilat - image)
plt.imshow(diff)
plt.title("bilat - diff")

plt.show()

