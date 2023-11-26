import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import scipy

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'C:\\Users\\Ethan\\Desktop\\Software Capstone\\Flower Classification V2\\V2\\Training Data'
validation_dir = 'C:\\Users\\Ethan\\Desktop\\Software Capstone\\Flower Classification V2\\V2\\Validation Data'
test_dir = 'C:\\Users\\Ethan\\Desktop\\Software Capstone\\Flower Classification V2\\V2\\Testing Data'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

# Get the number of classes
num_classes = len(train_generator.class_indices)

# Model building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')  # Set num_classes to the number of classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model training
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # Number of batches per epoch
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50  # Number of validation batches
)

model.save('model.h5')



