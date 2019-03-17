from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
import numpy as np
import cv2

def construct_model(model_directory):
    model = interpreter_wrapper.Interpreter(model_path=model_directory)
    model.allocate_tensors()
    print('Model Loaded')

    return model

def model_predict(img_path, model, target_size):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model.set_tensor(input_details[0]['index'], x)
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output)

    return prediction

def decode(preds):
    score = round(100 - float(preds)*100, 3)

    hotdogness = "Hot-Dog-ness: " + str(score)
    return hotdogness, score

def generate_overlay(img_path, preds, tick_path, cross_path, target_size):
    background = cv2.imread(img_path)
    tick = cv2.imread(tick_path)
    cross = cv2.imread(cross_path)

    background = cv2.resize(background, target_size)
    tick = cv2.resize(tick, target_size)
    cross = cv2.resize(cross, target_size)

    if preds > 75:
        result_image = cv2.addWeighted(background, 1.0, tick, 0.9, 0)
    else:
        result_image = cv2.addWeighted(background, 1.0, cross, 0.9, 0)

    cv2.imwrite(img_path, result_image)
