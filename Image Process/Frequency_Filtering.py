import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../Data/lenna.png', 0).astype(np.float32) / 255

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0,1])


rows, cols = image.shape
center_row = rows // 2
center_col = cols // 2


radius1 = int(input("Enter the radius of the inner circle: "))
radius2 = int(input("Enter the radius of the outer circle: "))


mask1 = np.zeros((rows, cols), dtype=np.uint8)
mask2 = np.zeros((rows, cols), dtype=np.uint8)
cv2.circle(mask1, (center_col, center_row), radius1, (1, 1, 1), -1)
cv2.circle(mask2, (center_col, center_row), radius2, (1, 1, 1), -1)

band_pass_mask = mask2 - mask1
filtered_shifted = fft_shift * band_pass_mask[:, :, np.newaxis]


fft_shift_inverse = np.fft.ifftshift(filtered_shifted, axes=[0, 1])

filtered_image = cv2.idft(fft_shift_inverse)
filtered_image = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])


plt.figure()
plt.subplot(121)
plt.axis('off')
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(122)
plt.axis('off')
plt.imshow(filtered_image, cmap='gray')
plt.title('Band Pass')

plt.show()