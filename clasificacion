import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

train_dir = os.path.join('train')

validation_dir = os.path.join('test')

train_green_dir = os.path.join(train_dir, 'green')
train_ground_dir = os.path.join(train_dir, 'ground')
train_leafless_dir = os.path.join(train_dir, 'leafless')
train_red_dir = os.path.join(train_dir, 'red')
train_yellow_dir = os.path.join(train_dir, 'yellow')

validation_green_dir = os.path.join(validation_dir, 'green')
validation_ground_dir = os.path.join(validation_dir, 'ground')
validation_leafless_dir = os.path.join(validation_dir, 'leafless')
validation_red_dir = os.path.join(validation_dir, 'red')
validation_yellow_dir = os.path.join(validation_dir, 'yellow')


num_green_tr = len(os.listdir(train_green_dir))
num_ground_tr = len(os.listdir(train_ground_dir))
num_leafless_tr = len(os.listdir(train_leafless_dir))
num_red_tr = len(os.listdir(train_red_dir))
num_yellow_tr = len(os.listdir(train_yellow_dir))



num_green_val = len(os.listdir(validation_green_dir))
num_ground_val = len(os.listdir(validation_ground_dir))
num_leafless_val = len(os.listdir(validation_leafless_dir))
num_red_val = len(os.listdir(validation_red_dir))
num_yellow_val = len(os.listdir(validation_yellow_dir))


total_train = num_green_tr + num_ground_tr + num_leafless_tr + num_red_tr + num_yellow_tr
total_val = num_green_val + num_ground_val + num_leafless_val + num_red_val + num_yellow_val

print('total de imaganes green de entrenamiento:', num_green_tr)
print('total de imaganes ground de entrenamiento:', num_ground_tr)
print('total de imaganes leafless de entrenamiento:', num_leafless_tr)
print('total de imaganes red de entrenamiento:', num_red_tr)
print('total de imaganes yellow de entrenamiento:', num_yellow_tr)

print('total de imagenes de validacion green:', num_green_val)
print('total de imagenes de validacion ground:', num_ground_val)
print('total de imagenes de validacion leafless:', num_leafless_val)
print('total de imagenes de validacion red:', num_red_val)
print('total de imagenes de validacion yellow:', num_yellow_val)


print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


batch_size = 32
epochs = 15
IMG_HEIGHT = 76
IMG_WIDTH = 76

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('on')
    plt.tight_layout()
    plt.show()


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

