import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []
image_names = ['boat1', 'budapest1', 'newspaper1', 's1']

for file in image_names:
    image = cv2.imread('../Data/stitching/' + file + '.jpg')

    if image is None:
        print('Load Fail')
    else:
        images.append(image)

for i, image in enumerate(images):
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges_Canny = cv2.Canny(blurred_image, 100, 200)

    gray_float = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    image_with_harris = image.copy()
    image_with_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]  # 코너를 빨간색으로 표시

    plt.figure(figsize=(15, 10))

    # 원본 이미지 표시
    plt.subplot(3, 1, 1)
    plt.imshow(original)
    plt.title(f'Original Image - {image_names[i]}')
    plt.axis('off')

    # Canny Edge Detection 결과 표시
    plt.subplot(3, 1, 2)
    plt.imshow(edges_Canny, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    # Harris Corner Detection 결과 표시
    plt.subplot(3, 1, 3)
    plt.imshow(cv2.cvtColor(image_with_harris, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corners')
    plt.axis('off')

    plt.show()