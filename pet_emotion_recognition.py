import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications import MobileNetV2

conv_base=MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

#conv_base.summary()

model=Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.summary()

conv_base.trainable=False

#from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

batch_size=32

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    '/kaggle/input/datasets/anshtanwar/pets-facial-expression-dataset/Master Folder/train',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator=test_datagen.flow_from_directory(
    '/kaggle/input/datasets/anshtanwar/pets-facial-expression-dataset/Master Folder/test',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical')

import matplotlib.pyplot as plt
x_batch, y_batch = next(train_generator)
plt.figure(figsize=(10, 6))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.show()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[early_stopping])

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

from tensorflow.keras.models import load_model
model.save('/kaggle/working/pet_emotion.h5')
model.save("/kaggle/working/pet_emotion.keras")

loss, accuracy = model.evaluate(validation_generator)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

validation_generator.class_indices

from keras.preprocessing import image
import numpy as np
model = load_model('/kaggle/working/pet_emotion.h5')

file_path = "/kaggle/input/datasets/anshtanwar/pets-facial-expression-dataset/Master Folder/test/happy/005.jpg"

img = image.load_img(file_path, target_size=(224,224))
x = image.img_to_array(img)
x=x/255
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
preds=np.argmax(preds, axis=1)
print(preds)

file_path = "/kaggle/input/datasets/anshtanwar/pets-facial-expression-dataset/Master Folder/test/happy/005.jpg"

img = image.load_img(file_path, target_size=(224, 224))
x = image.img_to_array(img)
x=x/255
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
preds=np.argmax(preds, axis=1)
if preds == 0:
    print("Angry")
elif preds == 1:
    print("Happy")
elif preds == 2:
    print("Relaxed")
else:
    print("Sad")
