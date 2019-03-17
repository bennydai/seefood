import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
from aux import *
from model import *

full_model = load_model(full_model_filename)
lite_model = interpreter_wrapper.Interpreter(model_path=lite_model_filename)

lite_model.allocate_tensors()

lite_input_details = lite_model.get_input_details()
lite_output_details = lite_model.get_output_details()

print('Models Loaded')

# Create generators
train_generator, test_generator, valid_generator = grab_generator(img_width, img_height,
                                                                  train_data_directory,
                                                                  test_data_directory,
                                                                  valid_data_directory)

list_of_images = os.listdir(predict_data_directory)

for num in range(0, len(list_of_images)):
    img = image.load_img(predict_data_directory + list_of_images[num],
                         target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array_expanded_dims)

    prediction = full_model.predict(processed_img)

    lite_model.set_tensor(lite_input_details[0]['index'], processed_img)
    lite_model.invoke()
    lite_output = lite_model.get_tensor(lite_output_details[0]['index'])

    full_model_output = np.squeeze(prediction)
    lite_output = np.squeeze(lite_output)

    list_of_preds = [full_model_output, lite_output]
    print(list_of_preds)

    full_prediction = round(100 - full_model_output*100, 3)
    lite_prediction = round(100 - lite_output*100, 3)

    name_of_file = list_of_images[num]

    hotdogness = ("HotDogNess: " + str(full_prediction) + " " + str(lite_prediction))
    title = name_of_file + ' ' + hotdogness

    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.savefig('results/' + name_of_file)
    plt.show()

score = full_model.evaluate_generator(test_generator, steps=nb_test_samples)
print(score)
