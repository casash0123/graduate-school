import cv2

images = []
image_gray = []
image_names = ['dog_a', 'dog_b']

for file in image_names:
    image = cv2.imread('../Data/stitching/' + file + '.jpg')

    if image is None:
        print('Load Fail')
    else:
        images.append(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray.append(gray)


feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(image_gray[0], mask=None, **feature_params)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p1, st, err = cv2.calcOpticalFlowPyrLK(image_gray[0], image_gray[1], p0, None, **lk_params)

good_new = p1[st == 1]
good_old = p0[st == 1]

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    images[1] = cv2.circle(images[1], (a, b), 5, (0, 255, 0), -1)
    images[1] = cv2.line(images[1], (a, b), (c, d), (255, 0, 0), 2)

cv2.imshow('Optical Flow', images[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
