from __future__ import print_function
import os

import keras
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
data_augmentation = False

epochs = 14

batch_size = 4
num_classes = 10
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model_1.h5'

# train test split
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_test.shape[0], 'test samples')
print(x_train.shape[0], 'train samples')
print('x_train shape:', x_train.shape)

# Convert class vectors to binary class matrices.
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-4)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if data_augmentation:
    print('Using real-time data augmentation.')
    #preprocessing fro realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,  
        samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06,  
        rotation_range=0, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.,  
        zoom_range=0., channel_shift_range=0., fill_mode='nearest', cval=0., horizontal_flip=True,  
        vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0)

    #feature-wise normalization.
    datagen.fit(x_train)

    # create models by datagen.flow()
    # then fit the models with the pre-trained parameters
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                        validation_data=(x_test, y_test), workers=4)

else:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
              shuffle=True)
    
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test accuracy in percentage: ', scores[1]*100)
print('Test loss: ', scores[0])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
