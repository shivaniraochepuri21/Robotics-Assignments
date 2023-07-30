import math
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()

print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

#--------------data visualisation------------------------------
# plot first few images of the training dataset
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(trainX[i])
pyplot.show()

# plot first few images of the testing dataset
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(testX[i])
pyplot.show()

#load the test_data batch of cifar10
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file1 = file1 = '/home/shivani/cifar-10-batches-py/test_batch'
batch1 = unpickle(file1)
data = []
labels = []
data.append(batch1[b'data'])
labels.append(batch1[b'labels'])
data = np.array(data)
labels = np.array(labels)
#rgb_data = np.concatenate(data)
#batch_labels = np.concatenate(labels) 

# dimensions of our images
img_width, img_height = 32, 32

# load the model that is pretrained i.e, 'keras_cifar10_trained_model_1.h5'

model = load_model('/home/shivani/saved_models/keras_cifar10_trained_model_1.h5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#-------------------------random test images other than those in cifar10, (from internet))
s = ['/home/shivani/Downloads/car1.jpg', '/home/shivani/Downloads/bird1.jpg', '/home/shivani/Downloads/ship1.jpg']

# predicting images
images = []
for i in range(3):
	img = image.load_img(s[i], target_size = (img_width, img_height))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	images = np.vstack([x])
	#images = np.concatenate(images,x, axis = 1)	
	classes = model.predict_classes(images, batch_size=10)
	#print(images.shape)
	#print (classes[0])

data = data[0]

classes_test = []
for i in range(10000):
	d = data[i,:].reshape((1,32,32,3))
	classes_test.append(model.predict_classes(d, batch_size=10))
	#print(len(d))
	#print("\n")
	#print(d)

print(len(classes_test))
print(classes_test)	

#print (classes)
#print (classes[0])
#print (classes[0][0])
