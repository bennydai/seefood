import tensorflow as tf

# tested on tensorflow-gpu 1.12 - if later, change tf.contrib.lite -> tf.lite
# quantization

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_file="../weights/super_hot_dog_200.h5",
	input_shapes={"input_1":[1,280,280,3]})
converter.post_training_quantize = True
quan_model = converter.convert()
open("../weights/super_hot_dog_200.tflite", "wb").write(quan_model)
