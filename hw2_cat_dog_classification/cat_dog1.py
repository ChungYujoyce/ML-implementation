'''
Reference: Kaggle
https://www.kaggle.com/bulentsiyah/dogs-vs-cats-classification-vgg16-fine-tuning
'''
import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import os
from tensorflow.keras.models import Model

base_dir = 'cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
batch_size=15

print('total training cat images:', len(os.listdir(train_cats_dir)))

def create_model():
    # Hint: model architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))

    #model.add(layers.BatchNormalization()) # add batch
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2)) # dropout prevent overfitting

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.BatchNormalization()) # add batch
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2)) # dropout

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.BatchNormalization()) # add batch
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2)) # dropout

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
            # Hint: optimizer & learning rate  
            optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['acc'])

    return model

model = create_model()
model.summary()

# Hint: data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')


callbacks = []
callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint.h5', save_best_only=True, save_weights_only=False))
total_validate = train_generator.__len__()
total_train = validation_generator.__len__()
history = model.fit_generator(
    train_generator, 
    epochs=50,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
) 

# Plot training history
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('acc.png')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.png')
