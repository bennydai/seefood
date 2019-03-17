import matplotlib.pyplot as plt
from aux import *
from model import *

# Generate model
model = create_model()

# Create generators
train_generator, test_generator, valid_generator = grab_generator(img_width, img_height,
                                                                  train_data_directory,
                                                                  test_data_directory,
                                                                  valid_data_directory)

# Train
callbacks_list = create_callback(full_model_filename)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_steps=nb_valid_samples,
    validation_data=valid_generator,
    callbacks=callbacks_list, shuffle=True)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
