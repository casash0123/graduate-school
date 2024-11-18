import cv2

images = []
image_name = 'boat'
count_stitch = 4

for i in range(count_stitch):
    image = cv2.imread('../Data/stitching/' + image_name + f'{i + 1}' + '.jpg')

    if image is None:
        print('Load Fail')
    else:
        images.append(image)

stitcher = cv2.createStitcher()

status, stitched_image = stitcher.stitch(images)

if cv2.Stitcher_OK == status :
    cv2.imshow('Stitched Image', stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
