import os
import shutil
import cv2

pascal_data_directory = '/Users/bennydai/VOC2007/JPEGImages/'
copied_over_data_directory = 'data/pascal_not_human/'

list_of_pascal_images = os.listdir(pascal_data_directory)
list_of_pascal_images.sort()

count = 0

for num in range(len(list_of_pascal_images)):
    img = cv2.imread(pascal_data_directory + list_of_pascal_images[num])
    cv2.imshow('image', img)
    cv2.waitKey(1)

    keep = input('Did you want this photo? ')
    if keep == 'Y' or keep == 'y':
        cv2.destroyAllWindows()
        count += 1
        shutil.copy(pascal_data_directory + list_of_pascal_images[num], copied_over_data_directory + 'pascal' + str(count) + '.jpg')
        print('Number of images coped: ' + str(count))
    else:
        cv2.destroyAllWindows()


