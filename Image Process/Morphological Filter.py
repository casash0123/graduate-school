import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../Data/lenna.png', 0)

selectedThreshold = int(input("1. Otsu, 2. Adaptive : "))

if 1 == selectedThreshold:
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresholdTitle = 'Otsu'

elif 2 == selectedThreshold:
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    thresholdTitle = 'Adaptive'

else :
    exit()



plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('Origine')
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.axis('off')
plt.title(thresholdTitle)
plt.imshow(mask, cmap='gray')

plt.show()


selectedMorphology = int(input("1. Erosion, 2. Dilation, 3. Opening, 4. Closeing :"))
selectedRepeat = int(input("Repeat : "))

if 1 == selectedMorphology:
    result = cv2.erode(image, None, iterations=selectedRepeat)
    selected_name = 'Erosion'

elif 2 == selectedMorphology:
    result = cv2.dilate(image, None, iterations=selectedRepeat)
    selected_name = 'Dilation'

elif selectedMorphology == 3:
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, None, iterations=selectedRepeat)
    selected_name = 'Opening'

elif selectedMorphology == 4:
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, None, iterations=selectedRepeat)
    selected_name = 'Closing'

else:
    exit()


plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('Origine')
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.axis('off')
plt.title(selected_name)
plt.imshow(result, cmap='gray')

plt.show()