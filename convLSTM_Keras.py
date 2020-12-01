#Rehaan Ahmad, Brian Yang
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import keras
from keras import backend as K
from keras import activations
from keras.optimizers import SGD
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import metrics
from keras.layers import Input, Embedding, Dense, Multiply
from keras.models import Model
from keras.layers.recurrent import _generate_dropout_mask
from keras.layers.recurrent import _standardize_args

import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
from keras.engine.base_layer import InputSpec, Layer
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.legacy.layers import Recurrent, ConvRecurrent2D
from keras.layers.recurrent import RNN
from keras.layers.convolutional_recurrent import ConvRNN2D, ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import transpose_shape


channels_img = 2
M = 100
N = 100
sequence_length = 50
TESTING_EXAMPLES = 56
TOTAL_DATA = 755
satellite_images = np.zeros([TOTAL_DATA, channels_img, M, N])
cloud_masks = np.zeros([TOTAL_DATA, M, N])

inputs_train = np.empty((TOTAL_DATA - sequence_length - TESTING_EXAMPLES, sequence_length, 2, 100, 100))
outputs_train = np.empty((TOTAL_DATA - sequence_length - TESTING_EXAMPLES, 1, 100, 100))
masks_train = np.empty((TOTAL_DATA - sequence_length - TESTING_EXAMPLES, 1, 100, 100))

inputs_test = np.empty((TESTING_EXAMPLES, sequence_length, 2, 100, 100))
outputs_test = np.empty((TESTING_EXAMPLES, 1, 100, 100))
masks_test = np.empty((TOTAL_DATA - sequence_length - TESTING_EXAMPLES, 1, 100, 100))

inputs_all = np.empty((TOTAL_DATA - sequence_length, sequence_length, 2, 100, 100))
outputs_all = np.empty((TOTAL_DATA - sequence_length, 1, 100, 100))
mask_all = np.empty((TOTAL_DATA - sequence_length, 1, 100, 100))



class ValidationLearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, verbose=1):
        super(ValidationLearningRateScheduler, self).__init__()
        self.verbose = verbose
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        self.val_losses.append(logs.get('val_root_mean_squared_error'))

        lr = float(K.get_value(self.model.optimizer.lr))
        if len(self.val_losses) > 1 and self.val_losses[-1] > self.val_losses[-2]:
            lr = lr * 0.8
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nEpoch %05d: ValidationLearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))




def normalize_np(arr):
    return (arr+0.3)/1.3

def generateCloudMask():
    global cloud_masks
    cloud_masks = np.zeros([TOTAL_DATA, M, N])
    totalNumber = 0
    for a in range(0, 1):
        for b in range(0, 1):
            counter = 0
            if os.path.isdir("cloudMasks"):
                filenames = []
                for filename in os.listdir("cloudMasks"): 
                    filenames.append(filename)
                filenames.sort()
                for filename in filenames:
                    arr = np.load("cloudMasks" + "/" + filename)
                    if counter<satellite_images.shape[0]:
                        #Make 0,1,2,3 cloud mask into a 0-1 cloud mask. 
                        arr[arr == 3] = 4
                        arr[arr == 2] = 4
                        arr[arr == 1] = 5
                        arr[arr == 0] = 5
                        arr[arr == -1] = 0
                        arr[arr == 4] = 0
                        arr[arr == 5] = 1
                        cloud_masks[totalNumber] = arr[0:100, 0:100]
                        print("cloud mask: " + str(np.mean(cloud_masks[totalNumber])))
                    counter += 1
                    totalNumber += 1
                
    

def loadData():
    global satellite_images

    satellite_images = np.zeros([TOTAL_DATA, channels_img, M, N])

    dataNDVI = np.zeros([14])
    dataRain = np.zeros([10])

    ndvi_images = np.zeros([TOTAL_DATA, M, N])
    rain_images = np.zeros([TOTAL_DATA, M, N])

    totalNumber = 0
    for a in range(0, 1):
        for b in range(0, 1):
            counter = 0
            if os.path.isdir("combineUruguay"):
                filenames = []
                for filename in os.listdir("combineUruguay"): 
                    filenames.append(filename)
                filenames.sort()
                for filename in filenames:
                    arr = np.load("combineUruguay" + "/" + filename)
                    if counter<satellite_images.shape[0]:
                        satellite_images[counter][0] = normalize_np(arr[0][0:100, 0:100])
                        satellite_images[counter][1] = arr[1][0:100, 0:100]/214
                        print("read in data")
                        ndvi_images[totalNumber] = normalize_np(arr[0][0:100, 0:100])
                        rain_images[totalNumber] = arr[1][0:100, 0:100]/214
                    counter += 1
                    totalNumber += 1
                
    print("AQUA Data Loaded")

    print(counter)

    print(np.mean(ndvi_images))
    print(np.mean(rain_images))

    print(np.std(ndvi_images))
    print(np.std(rain_images))

    print(np.min(ndvi_images))
    print(np.min(rain_images))

    print(np.max(ndvi_images))
    print(np.max(rain_images))

    return totalNumber


def prepareData():

    global inputs_all
    global outputs_all
    global mask_all

    inputs_all = np.empty((TOTAL_DATA - sequence_length, sequence_length, 2, 100, 100))
    outputs_all = np.empty((TOTAL_DATA - sequence_length, 1, 100, 100))
    mask_all = np.empty((TOTAL_DATA - sequence_length, 1, 100, 100))

    for i in range(TOTAL_DATA - sequence_length):
        for j in range(sequence_length):
            inputs_all[i][j] = satellite_images[i + j]

    for i in range(TOTAL_DATA - sequence_length):
        outputs_all[i] = satellite_images[i + sequence_length][0:1]
        mask_all[i] = cloud_masks[i + sequence_length].reshape(1, M, N)
    



def splitData(count):

    global inputs_train
    global outputs_train
    global masks_train

    global inputs_test
    global outputs_test
    global masks_test

    inputs_train = np.empty((TOTAL_DATA - sequence_length - count, sequence_length, 2, 100, 100))
    outputs_train = np.empty((TOTAL_DATA - sequence_length - count, 1, 100, 100))
    masks_train = np.empty((TOTAL_DATA - sequence_length - count, 1, 100, 100))

    inputs_test = np.empty((count, sequence_length, 2, 100, 100))
    outputs_test = np.empty((count, 1, 100, 100))
    masks_test = np.empty((TOTAL_DATA - sequence_length - count, 1, 100, 100))

    c = random.sample(range(0, TOTAL_DATA - sequence_length), count)

    train_counter = 0
    test_counter = 0

    for i in range(TOTAL_DATA - sequence_length):
        if i in c:
            inputs_test[test_counter] = inputs_all[i]
            outputs_test[test_counter] = outputs_all[i]*mask_all[i]
            masks_test[test_counter] = mask_all[i]
            test_counter += 1
        else:
            inputs_train[train_counter] = inputs_all[i]
            outputs_train[train_counter] = outputs_all[i]*mask_all[i]
            masks_train[train_counter] = mask_all[i]
            print("got here 2: " + str(np.mean(masks_train[train_counter])) + " : " + str(np.mean(mask_all[i])))
            train_counter += 1


def root_mean_squared_error(y_true, y_pred):
    nonzero = K.tf.count_nonzero(y_pred)
    return K.switch(K.equal(nonzero,0)
                    , K.constant(value=0.)
                    , K.sqrt(K.sum(K.square(y_pred - y_true))/tf.cast(nonzero, tf.float32)))
def mean_squared_error_loss(y_true, y_pred):
    nonzero = K.tf.count_nonzero(y_pred)
    #return K.sum(K.square(y_pred - y_true))/tf.cast(tf.multiply(nonzero, tf.constant(2)), tf.float32)
    return K.sum(K.square(y_pred - y_true))/tf.cast(tf.constant(2), tf.float32)

generateCloudMask()

print(loadData())
prepareData()
splitData(TESTING_EXAMPLES)

print(inputs_train.shape)
print(outputs_train.shape)
print(inputs_test.shape)
print(outputs_test.shape)

mc = keras.callbacks.ModelCheckpoint('modelsPerEpoch/weights{epoch:06d}.hdf5', 
                                     save_weights_only=False, 
                                     period=1)

decay_learner = ValidationLearningRateScheduler()

main_input = Input(shape=(None, 2, 100, 100), dtype='float32', name='input')

mask=Input(shape=(1, 100, 100), dtype='float32', name='mask')

hidden = ConvLSTM2D(filters=16, 
                    kernel_size=(5, 5),  
                    padding='same',  
                    return_sequences=False, 
                    data_format='channels_first')(main_input)

output = Conv2D(filters=1, 
                kernel_size=(1, 1), 
                padding='same',
                activation='sigmoid',
                kernel_initializer='glorot_uniform',
                data_format='channels_first',
                name='output')(hidden)

output_with_mask=Multiply()([output, mask])

sgd = SGD(lr=0.002, momentum=0.0, decay=0.0, nesterov=False)

model = Model(inputs=[main_input, mask], outputs=output_with_mask)

model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=[metrics.mse, root_mean_squared_error])

print(inputs_train.shape)
print(masks_train.shape)
print(np.mean(inputs_train))
print(np.mean(masks_train))

for i in range(0, len(masks_train)):
    print(np.mean(masks_train[i]))

training_data = model.fit([inputs_train, masks_train],
		                      outputs_train,
		                      epochs=20,
                          batch_size=1,
                          validation_split=0.22,
                          verbose=1,
                          callbacks=[mc, decay_learner],
                          shuffle=True)

score = model.evaluate([inputs_test, masks_test], outputs_test, batch_size=1)
predictions = model.predict(inputs_test)

np.save('test_arrays', predictions)
np.save('training_loss', training_data.history['root_mean_squared_error'])
np.save('validation_loss', training_data.history['val_root_mean_squared_error'])

print(score)

plt.plot(training_data.history['root_mean_squared_error'])
plt.plot(training_data.history['val_root_mean_squared_error'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
