from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from aux import *

def create_model():
    base_model = InceptionV3(weights = 'imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    model.summary()

    layer_index = int(0.75 * len(model.layers))

    for layer in model.layers[:layer_index]:
        layer.trainable = False
    for layer in model.layers[layer_index:]:
        layer.trainable = True

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

    return model

def grab_generator(img_width, img_height, train_data_directory, test_data_directory, valid_data_directory):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.5,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10,
        brightness_range=[0.5, 1.5]
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    valid_generator = test_datagen.flow_from_directory(
        valid_data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    print('Train generator' + str(list(train_generator.class_indices.keys())))
    print('Test generator' + str(list(test_generator.class_indices.keys())))
    print('Valid generator' + str(list(valid_generator.class_indices.keys())))

    return train_generator, test_generator, valid_generator

def create_callback(model_filename):
    checkpoint = ModelCheckpoint(model_filename, monitor='val_acc',
                                 verbose=1, save_best_only=True, save_weights_only=False, mode='max')

    tb_checkpoint = TensorBoard(log_dir='./logs')
    callbacks_list = [checkpoint, tb_checkpoint]
    return callbacks_list
