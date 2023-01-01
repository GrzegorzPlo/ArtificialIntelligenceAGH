import tensorflow as tf
import os

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

batch_size = 32
img_height = 224
img_width = 224

data_dir = "data"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, 
    image_size=(img_height, img_width),
    validation_split=0.2,
    subset="training",
    seed=0xdeadbeef,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, 
    image_size=(img_height, img_width),
    validation_split=0.2,
    subset="validation",
    seed=0xdeadbeef,
    batch_size=batch_size
)


model = models.Sequential()

model.add(layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))

model.add(layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(units=4096,activation="relu"))
model.add(layers.Dense(units=4096,activation="relu"))
model.add(layers.Dense(units=10, activation="softmax"))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

epochs = 25

history = model.fit(train_ds, epochs=25, validation_data=val_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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

model.summary()

model.save('grpl_vgg_16_for_batch_32.h5')
print('Model Saved!')