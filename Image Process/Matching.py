import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []
image_names = ['../Data/stitching/boat1.jpg', '../Data/stitching/budapest1.jpg', '../Data/stitching/newspaper1.jpg', '../Data/stitching/s1.jpg']

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

keypoints_list = []
descriptors_list = []
valid_images = []

for file in image_names:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Failed, Load Image:{file}")
        continue

    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)
    valid_images.append(file)

def find_homography(src_pts, dst_pts):
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def match_and_find_homography(des1, des2, kp1, kp2, method):
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H = find_homography(src_pts, dst_pts)
    return H

def warp_image(img1, img2, H):
    h, w = img1.shape
    result = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    return result

# 이미지 선택
img_idx1 = 0
img_idx2 = 1

if img_idx1 >= len(keypoints_list) or img_idx2 >= len(keypoints_list):
    print("Error, Index")
    exit()

H_sift = match_and_find_homography(descriptors_list[img_idx1], descriptors_list[img_idx2],
                                   keypoints_list[img_idx1], keypoints_list[img_idx2], 'SIFT')

img1 = cv2.imread(valid_images[img_idx1])
img2 = cv2.imread(valid_images[img_idx2])
if img1 is None or img2 is None:
    print("Failed, Load Image")
    exit()

result = warp_image(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), H_sift)

# 결과 표시
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.title('Original Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Warped Image')
plt.axis('off')

plt.show()
