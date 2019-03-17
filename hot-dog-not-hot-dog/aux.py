from keras import backend as KBackend
import cv2
import os
import numpy as np

def determine_number_of_files(directory):
    list_of_hotdogs = os.listdir(directory + 'hot_dog/')
    list_of_nothotdogs = os.listdir(directory + 'not_hot_dog/')

    list_of_files = list_of_hotdogs + list_of_nothotdogs
    list_of_images = []

    for names in list_of_files:
        if names.endswith('.jpg'):
            list_of_images.append(names)
    return len(list_of_images)

def sample_image_size(directory):
    hot_dog_directory = directory + 'hot_dog/'
    list_of_hotdog = os.listdir(hot_dog_directory)
    for num in range(0, len(list_of_hotdog), 20):
        img = cv2.imread(hot_dog_directory + list_of_hotdog[num])
        height, width, channels = img.shape
        print(height, width, channels)

    not_hot_dog_directory = directory + 'not_hot_dog/'
    list_of_nothotdogs = os.listdir(not_hot_dog_directory)
    for num in range(0, len(list_of_nothotdogs), 20):
        img = cv2.imread(not_hot_dog_directory + list_of_nothotdogs[num])
        height, width, channels = img.shape
        print(height, width, channels)

# parameters
train_data_directory = 'data/train/'
test_data_directory = 'data/test/'
valid_data_directory = 'data/valid/'
predict_data_directory = 'data/predict/'
full_model_filename = '../weights/super_hot_dog_200.h5'
lite_model_filename = '../weights/super_hot_dog_200.tflite'

batch_size = 64
epochs = 200
img_width, img_height = 280, 280

train_samples = determine_number_of_files(train_data_directory)
test_samples = determine_number_of_files(test_data_directory)
valid_samples = determine_number_of_files(valid_data_directory)
nb_train_samples = train_samples // batch_size
nb_test_samples = 1
nb_valid_samples = valid_samples // batch_size

if KBackend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
