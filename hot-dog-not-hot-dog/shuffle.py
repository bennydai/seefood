import os
import shutil
from random import shuffle

train_directory = 'data/train/'
test_directory = 'data/test/'
valid_directory = 'data/valid/'

hot_dog = 'hot_dog/'
not_hot_dog = 'not_hot_dog/'

hot_dog_directory = 'data/hot_dog/'
not_hot_dog_directory = 'data/not_hot_dog/'

hot_dog_list = os.listdir(hot_dog_directory)
not_hot_dog_list = os.listdir(not_hot_dog_directory)
shuffle(hot_dog_list)
shuffle(not_hot_dog_list)

print(len(hot_dog_list))
print(len(not_hot_dog_list))


def random_split(class_directory, list_of_images, object, train_directory, test_directory, valid_directory):
	train_index = int(0.89 * len(list_of_images))
	valid_index = int(0.10 * len(list_of_images))

	for num in range(len(list_of_images)):
		if num < train_index:
			shutil.copy(class_directory + list_of_images[num],
						train_directory + object + list_of_images[num])
		elif num < train_index + valid_index:
			shutil.copy(class_directory + list_of_images[num],
						valid_directory + object + list_of_images[num])
		else:
			shutil.copy(class_directory + list_of_images[num],
						test_directory + object + list_of_images[num])


random_split(hot_dog_directory, hot_dog_list, hot_dog, train_directory, test_directory, valid_directory)
random_split(not_hot_dog_directory, not_hot_dog_list, not_hot_dog, train_directory, test_directory, valid_directory)

print('All images have been shuffled')
