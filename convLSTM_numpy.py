# Rehaan, Brian
# The following code is a numpy (cupy for gpu) implementation of a ConvLSTM built
# for forecasting the next image in a sequence of antecedent NDVI and rain. The
# ConvLSTM also takes in a cloudmasks, and only trained on sufficiently high 
# quality pixels. If it is a particularly cloudy day, the ConvLSTM will simply
# reconstruct the data by using its forecast from antecedent data and replace 
# the low quality image.

import math
import chainer
import numpy as np
import chainer.functions as F
import copy
import cupy as cp
import sys
import random
import matplotlib.pyplot as plt        
import urllib
import zipfile
import os
from scipy.interpolate import UnivariateSpline

np.set_printoptions(threshold=np.inf)

# number of sets of images
S = 10

# number of images per sequence/set
T = 410

# dimensions of the image
M = 100
N = 100

channels_img = 2  # antecedent NDVI and rain
channels_hidden = 16
kernel_dimension = 5
pad_constant = 2

loss_clip_constant = 12

ndviMean = 0.66673688145266
ndviStdDev = 0.16560766237944935
rainMean = 0.19636724781555773
rainStdDev = 0.16560766237944935

# ndviMean = 0.10724276129701694
# rainMean = 0.018842877488562778
# ndviStdDev =  0.03155192804492544
# rainStdDev = 0.023218907256230225
TOTAL_DATA = 755
usableData = 0

satellite_images = np.empty([S, 711, channels_img, M, N])
learning_window = 50
cloud_masks = np.zeros([TOTAL_DATA, M, N])

prev_validate = 100
clip_threshold = 5
clip_threshold_output = 1
IMAGE_RECONSTRUCT = False

#-----------------------------------GLOROT INTIALIZATION------------------------------  
r_kernel_tanh = math.sqrt(6/((channels_hidden+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden))
r_kernel_sigmoid = math.sqrt(6/((channels_hidden+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden))
r_connected_weights = .6*math.sqrt(6/(channels_hidden + 1))

a_kernel = cp.random.uniform(-r_kernel_tanh, r_kernel_tanh, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
i_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
f_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
o_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
connected_weights = cp.random.normal(-r_connected_weights, r_connected_weights, (1, channels_hidden))
main_kernel = cp.concatenate((i_kernel, f_kernel, a_kernel, o_kernel))


#--------------------------------------BIAS INITIALIZATION-----------------------------
bias_c = cp.zeros([channels_hidden, M, N])
bias_i = cp.zeros([channels_hidden, M, N])
bias_f = cp.ones([channels_hidden, M, N])
bias_o = cp.zeros([channels_hidden, M, N])
bias_y = cp.zeros([channels_img, M, N])

learning_rate = 0.002
learning_rate_counter = 0
PRELOAD_SAVED_WEIGHTS = False

#--------------File Paths--------------
CLOUD_MASK_ROOT_FOLDER = "cloudMasks"
DATA_ROOT_FOLDER = "combineUruguay"

#--------------data structure used allowing to us to process data from both Terra and Aqua satellites--------------
class ImageSat(object):
    index = 0
    satellite = "SAME"

    def __init__(self, index, satellite):
        self.index = index
        self.satellite = satellite

def make_ImageSat(index, satellite):
    imageSat = ImageSat(index, satellite)
    return imageSat

#-----------helper functions for foreward prop and computing gradients in backprop----------  
def sigmoid(k):
    return 1 / (1 + cp.exp(-k))

def sigmoid_derivative(k):
    return sigmoid(k) * (1 - sigmoid(k))

def bipolar_sigmoid(k):
    return 2 / (1 + cp.exp(-k)) - cp.ones(k.shape)

def bipolar_derivative(k):
    return (1 - (bipolar_sigmoid(k)**2))/2

def tanh(k):
    return cp.tanh(k)

def tanh_derivative(k):
    return 1 - (tanh(k))**2

def expdecay(x):
    return distance/(1+cp.exp(-0.04*x))

def rect_linear_exponential(arr):
    arr2 = copy.deepcopy(arr)
    arr2 = expdecay(arr2)
    return arr2

def normalize_np(arr, mean, stddev):
    return (arr+0.3)/1.3
    #return 1/(1+np.exp(-(arr-mean)/stddev))

def unnormalize_np(arr, mean, stddev):
    return arr*1.3 - 0.3
    #return mean - stddev*np.log((1-arr)/arr)

def normalize_cp(arr, mean, stddev):
    return (arr+0.3)/1.3
    #return 1/(1+cp.exp(-(arr-mean)/stddev))

def unnormalize_cp(arr, mean, stddev):
    return arr*1.3 - 0.3
    #return mean - stddev*cp.log((1-arr)/arr)

def rect_linear_exponential_derivative(arr):
    arr2 = copy.deepcopy(arr)
    derivatives = 0.04*115*cp.exp(-0.04*arr2)/((1+cp.exp(-0.04*arr2))**2)
    return derivatives

def rect_linear(arr):
    newArr = copy.deepcopy(arr)
    newArr[arr<0] = 0
    return newArr

def rect_linear_derivative(arr):
    newArr = cp.zeros(arr.shape)
    newArr[arr>0] = 1
    return newArr

# x[t] is the input at time t
def forward_prop(x,local_time, currentIndex):
    global cloud_masks
    for t in np.arange(local_time):
        print(np.mean(cloud_masks[currentIndex + t]))
        if np.mean(cloud_masks[currentIndex + t]) < 0.82 and IMAGE_RECONSTRUCT == True and (currentIndex + t - learning_window) > 0:
            print("-------------------------------------HIGH CLOUD DENSITY...----------------------------------")
            print("------------------------------------RECONSTRUCTING IMAGE...---------------------------------")
            prediction6, pre_sigmoid_prediction6, hidden_prediction6, i6, f6, a6, c6, o6, h6 = forward_prop(cp.asarray(satellite_images[0][currentIndex + t - learning_window:currentIndex + t]), local_time, currentIndex + t - learning_window)
            x[t] = prediction6

    # Input Gate
    i = cp.empty([local_time, channels_hidden, M, N])

    # Forget Gate
    f = cp.empty([local_time, channels_hidden, M, N])

    # Memory
    a = cp.empty([local_time, channels_hidden, M, N])

    # Cell Gate
    c = cp.empty([local_time + 1, channels_hidden, M, N])
    c[-1] = cp.zeros([channels_hidden, M, N])
    # Output Gate
    o = cp.empty([local_time, channels_hidden, M, N])

    # Hidden Unit
    h = cp.empty([local_time + 1, channels_hidden, M, N])
    h[-1] = cp.zeros([channels_hidden, M, N])
    # LSTM FORWARD PROPAGATION
    for t in np.arange(local_time):
        temporary = cp.concatenate((x[t], h[t - 1]), axis=0)
        temporary = temporary.reshape(1, channels_img + channels_hidden, M, N)

        i[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernel[0:channels_hidden], b=None, pad=pad_constant)[0].data) + bias_i)

        f[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernel[channels_hidden:2*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_f)

        a[t] = tanh(cp.asarray(F.convolution_2d(temporary, main_kernel[2*channels_hidden:3*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_c)

        c[t] = cp.multiply(f[t], c[t - 1]) + cp.multiply(i[t], a[t])

        o[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernel[3*channels_hidden:4*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_o)

        h[t] = cp.multiply(o[t], tanh(c[t]))

    # 1 x 1 convolution
    #output = cp.matmul(connected_weights, h[local_time-1].reshape(channels_hidden, M * N)).reshape(M, N) + bias_y[0]
    output = cp.asarray(F.convolution_2d(h[local_time-1].reshape(1, channels_hidden, M, N), connected_weights.reshape(1, channels_hidden, 1, 1), b = None, pad = 0)[0][0].data) + bias_y[0]
    print("CONNECTED_WEIGHTS NORM: " + str(cp.linalg.norm(connected_weights)))
    print("HIDDEN_PREDICTION NORM: " + str(cp.linalg.norm(h[local_time-1])))
    print("CONNECTED_WEIGHTS MEAN: " + str(cp.mean(cp.abs(connected_weights))))
    print("HIDDEN_PREDICTION MEAN: " + str(cp.mean(cp.abs(h[local_time-1]))))
    true_output = sigmoid(output)
    return true_output, output, cp.reshape(h[local_time-1], (channels_hidden, M*N)), i, f, a, c, o, h



def calculate_loss2(prediction, y):
  #  prediction[prediction<0.1] = 0.00000001
    return -np.sum(np.multiply(y, np.log(prediction)) + np.multiply(np.ones(y.shape) - y, np.log(np.ones(y.shape) - prediction)))

#root mean square error
def rootmeansquare(prediction, y):
    return cp.sqrt(cp.sum((prediction - y)**2)/(np.count_nonzero(cp.asnumpy(prediction))))

# Calculate loss.
#loss function is MSE, since we are comparing two images. 
def calculate_loss(prediction, y):
    lossExpression = 0.5*cp.sum((prediction - y)**2)
    return lossExpression

def calculate_loss_modified(prediction, y):
    prediction[prediction == 0] = 0.00000001
    y[y == 0] = 0.00000001
    lossExpression = -cp.sum(cp.multiply(y, cp.log(prediction)) + cp.multiply(cp.ones(y.shape) - y, cp.log(cp.ones(y.shape) - prediction)))
    return lossExpression

def return_forecast(x, local_time, currentIndex):
    a,b,c,d,e,f,g,h,i = forward_prop(cp.asarray(x), local_time, currentIndex)
    return cp.asnumpy(a)

def loss_derivative(x, y):
    return (x-y)
  
#backpropagation through time (bptt) algorithm.
def bptt(x2, y2, iteration, local_time, currentIndex):
    x = cp.asarray(x2)
    y = cp.asarray(y2)

    global connected_weights
    global main_kernel
    global bias_i
    global bias_f
    global bias_c
    global bias_o
    global bias_y

    global learning_rate
    global learning_rate_counter

    # Perform forward prop
    prediction, pre_sigmoid_prediction, hidden_prediction, i, f, a, c, o, h = forward_prop(x, local_time, currentIndex)

    predictionLoss = unnormalize_cp(prediction, 0, 0)
    outputLoss = unnormalize_cp(y[0], 0, 0)

    prediction = prediction*cp.asarray(cloud_masks[currentIndex+local_time])
    y[0] = y[0]*cp.asarray(cloud_masks[currentIndex+local_time])
    predictionLoss = predictionLoss*cp.asarray(cloud_masks[currentIndex+local_time])
    outputLoss = outputLoss*cp.asarray(cloud_masks[currentIndex+local_time])

    loss = calculate_loss(predictionLoss, outputLoss)
    print("LOSS BEFORE: ")
    print(loss)

    # Calculate loss with respect to final layer
    dLdy_2 = loss_derivative(prediction, y[0])

    f2 = open("runtimedata/normlossderivative.txt", "a")
    f2.write(str(cp.linalg.norm(dLdy_2)) + "\n")
    # Calculate loss with respect to pre sigmoid layer
    dLdy_1 = cp.multiply(sigmoid_derivative(pre_sigmoid_prediction), dLdy_2)
    
    # Calculate loss with respect to last layer of lstm
    dLdh = cp.asarray(F.convolution_2d(dLdy_1.reshape(1, 1, M, N), (connected_weights.reshape(1, channels_hidden, 1, 1)).transpose(1,0,2,3), b=None, pad=0)[0].data)
    dLdw_0 = cp.asarray(F.convolution_2d(hidden_prediction.reshape(channels_hidden, 1, M, N), dLdy_1.reshape(1, 1, M, N), b=None, pad=0).data).transpose(1,0,2,3)
    dLdb_y = dLdy_1
    dLdw_0 = dLdw_0.reshape(1, channels_hidden)

    # uncomment code below if you would like to gradient clip the output layer.
    # if cp.linalg.norm(dLdw_0) > clip_threshold_output:
    #    dLdw_0 = dLdw_0*clip_threshold_output/cp.linalg.norm(dLdw_0)
    # if cp.linalg.norm(dLdb_y) > clip_threshold_output:
    #    dLdb_y = dLdb_y*clip_threshold_output/cp.linalg.norm(dLdb_y)
    
    #--------------------fully connected------------------
    bias_y = bias_y - learning_rate*dLdb_y
    connected_weights = connected_weights - learning_rate*dLdw_0

    # Initialize weight matrix
    dLdW = cp.zeros([4*channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension])

    # initialize biases
    dLdb_c = cp.zeros([channels_hidden, M, N])
    dLdb_i = cp.zeros([channels_hidden, M, N])
    dLdb_f = cp.zeros([channels_hidden, M, N])
    dLdb_o = cp.zeros([channels_hidden, M, N])

    # Initialize cell matrix
    dLdc_current = cp.zeros([channels_hidden, M, N])

    for t in cp.arange(local_time - 1, -1, -1):
        dLdo = cp.multiply(dLdh, tanh(c[t]))
        dLdc_current += cp.multiply(cp.multiply(dLdh, o[t]), (cp.ones((channels_hidden, M, N)) - cp.multiply(tanh(c[t]), tanh(c[t]))))
        dLdi = cp.multiply(dLdc_current, a[t])
        dLda = cp.multiply(dLdc_current, i[t])
        dLdf = cp.multiply(dLdc_current, c[t - 1])

        dLdc_previous = cp.multiply(dLdc_current, f[t])

        dLda = cp.multiply(dLda, (cp.ones((channels_hidden, M, N)) - cp.multiply(a[t], a[t]))) #dLda_hat

        dLdi = cp.multiply(cp.multiply(dLdi, i[t]), cp.ones((channels_hidden, M, N)) - i[t]) #dLdi_hat

        dLdf = cp.multiply(cp.multiply(dLdf, f[t]), cp.ones((channels_hidden, M, N)) - f[t]) #dLdf_hat

        dLdo = cp.multiply(cp.multiply(dLdo, o[t]), cp.ones((channels_hidden, M, N)) - o[t]) #dLdo_hat


        # CONCATENATE Z IN THE RIGHT ORDER SAME ORDER AS THE WEIGHTS
        dLdz_hat = cp.concatenate((dLdi, dLdf, dLda, dLdo), axis = 0) 

        #determine convolution derivatives
        #here we will use the fact that in z = w * I, dLdW = dLdz * I
        temporary = cp.concatenate((x[t], h[t - 1]), axis=0).reshape(channels_hidden + channels_img, 1, M, N)
        dLdI = cp.asarray(F.convolution_2d(dLdz_hat.reshape(1, 4*channels_hidden, M, N), main_kernel.transpose(1, 0, 2, 3), b=None, pad=pad_constant)[0].data) # reshape into flipped kernel dimensions
        dLdW_temp = cp.asarray((F.convolution_2d(temporary, dLdz_hat.reshape(4*channels_hidden, 1, M, N), b=None, pad=pad_constant).data).transpose(1,0,2,3)) #reshape into kernel dimensions
 
        # accumulate derivatives of weights and biases
        dLdW += dLdW_temp 
        dLdb_c += dLda
        dLdb_i += dLdi
        dLdb_f += dLdf
        dLdb_o += dLdo

        # reinitialize what you're passing back
        dLdh = dLdI[channels_img: channels_img+channels_hidden] 
        dLdc_current = dLdc_previous

    # #Clip all gradients again
    if cp.linalg.norm(dLdW) > clip_threshold:
        dLdW = dLdW*clip_threshold/cp.linalg.norm(dLdW)
    if cp.linalg.norm(dLdb_c) > clip_threshold:
        dLdb_c = dLdb_c*clip_threshold/cp.linalg.norm(dLdb_c)
    if cp.linalg.norm(dLdb_i) > clip_threshold:
        dLdb_i = dLdb_i*clip_threshold/cp.linalg.norm(dLdb_i)
    if cp.linalg.norm(dLdb_f) > clip_threshold:
        dLdb_f = dLdb_f*clip_threshold/cp.linalg.norm(dLdb_f)
    if cp.linalg.norm(dLdb_o) > clip_threshold:
        dLdb_o = dLdb_o*clip_threshold/cp.linalg.norm(dLdb_o)

    #---------------------update main kernel---------
    main_kernel = main_kernel - learning_rate*dLdW
    #--------------------update bias c-----------------------
    bias_c = bias_c - learning_rate*dLdb_c
    #--------------------update bias i-----------------------
    bias_i = bias_i - learning_rate*dLdb_i
    #--------------------update bias f-----------------------
    bias_f = bias_f - learning_rate*dLdb_f
    #--------------------update bias c-----------------------
    bias_o = bias_o - learning_rate*dLdb_o

    prediction2, pre_sigmoid_prediction2, hidden_prediction2, i2, f2, a2, c2, o2, h2 = forward_prop(x, local_time, currentIndex)
    prediction3 = prediction2*cp.asarray(cloud_masks[currentIndex + local_time])
    loss2 = calculate_loss(prediction3, y[0])

    prediction2 = unnormalize_cp(prediction2, ndviMean, ndviStdDev)
    prediction2 = prediction2*cp.asarray(cloud_masks[currentIndex + local_time])
    outputArr = unnormalize_cp(y[0], ndviMean, ndviStdDev)
    outputArr = outputArr*cp.asarray(cloud_masks[currentIndex + local_time])
    rms3 = rootmeansquare(prediction2, outputArr)

    f2 = open("runtimedata/loss.txt", "a")
    f2.write(str(rms3) + "\n")
    

    if loss2 > loss:
        #sys.exit("what")
        f2 = open("runtimedata/closeResults.txt", "a")
        f2.write(str(iteration))
        f2.write("\n")
        learning_rate_counter += 1
        if learning_rate_counter == 1:
            learning_rate_counter = 0
            #learning_rate = learning_rate*0.9
        print("----------------close------------------------------")


    print("backpropagation complete")

def generateCloudMask():
    global cloud_masks
    totalNumber = 0
    for a in range(0, 1):
        for b in range(0, 1):
            counter = 0
            if os.path.isdir(CLOUD_MASK_ROOT_FOLDER):
                filenames = []
                for filename in os.listdir(CLOUD_MASK_ROOT_FOLDER): 
                    filenames.append(filename)
                filenames.sort()
                for filename in filenames:
                    arr = np.load(CLOUD_MASK_ROOT_FOLDER + "/" + filename)
                    if counter<satellite_images.shape[1]:
                        #Make 0,1,2,3 cloud mask into a 0-1 cloud mask. 
                        arr[arr == 3] = 4
                        arr[arr == 2] = 4
                        arr[arr == 1] = 5
                        arr[arr == 0] = 5
                        arr[arr == -1] = 0
                        arr[arr == 4] = 0
                        arr[arr == 5] = 1
                        cloud_masks[totalNumber] = arr[0:100, 0:100]
                        print(np.mean(cloud_masks[totalNumber]))
                    counter += 1
                    totalNumber += 1


def loadData():
    generateCloudMask()

    global satellite_images

    totalNumber = 0
    for a in range(0, 1):
        for b in range(0, 1):
            counter = 0
            if os.path.isdir(DATA_ROOT_FOLDER):
                filenames = []
                for filename in os.listdir(DATA_ROOT_FOLDER): 
                    filenames.append(filename)
                filenames.sort()
                for filename in filenames:
                    arr = np.load(DATA_ROOT_FOLDER + "/" + filename)
                    if counter<satellite_images.shape[1]:
                        satellite_images[0][counter][0] = normalize_np(arr[0][0:100, 0:100], ndviMean, ndviStdDev)
                        satellite_images[0][counter][1] = arr[1][0:100, 0:100]/214
                        counter += 1

                    totalNumber += 1
            
    list1 = produceRandomImageArray()

    main(list1)

def MAPE(correct, prediction):
    return np.sum(np.absolute(correct-prediction)/correct)/100
    
def main(indexGeneralList):
    #initiate training process etc
    global stdev
    global mean
    global learning_rate

    global connected_weights
    global main_kernel
    global bias_i
    global bias_f
    global bias_c
    global bias_o
    global bias_y

    if PRELOAD_SAVED_WEIGHTS == True:
        connected_weights = cp.asarray(np.load('runtimedata/epoch18/5connected_weightsfinal3.npy'))
        main_kernel = cp.asarray(np.load('runtimedata/epoch18/5main_kernelfinal3.npy'))
        bias_y = cp.asarray(np.load('runtimedata/epoch18/5bias_yfinal3.npy'))
        bias_o = cp.asarray(np.load('runtimedata/epoch18/5bias_ofinal3.npy'))
        bias_c = cp.asarray(np.load('runtimedata/epoch18/5bias_cfinal3.npy'))
        bias_f = cp.asarray(np.load('runtimedata/epoch18/5bias_ffinal3.npy'))
        bias_i = cp.asarray(np.load('runtimedata/epoch18/5bias_ifinal3.npy'))

    global usableData
    usableData = len(indexGeneralList)

    indexList = indexGeneralList[0:int(0.7*usableData)] 
    validateList = indexGeneralList[(int(0.7*usableData)+1):int(0.9*usableData)]
    testList = indexGeneralList[(int(0.9*usableData)+1):usableData]
    
    f2 = open("runtimedata/indexListNumbers.txt", "a")
    for k in range(0, len(indexList)):
        f2.write(str(indexList[k].index) + " : " + str(indexList[k].satellite))
        f2.write("\n")

    f2 = open("runtimedata/validateListNumbers.txt", "a")
    for k in range(0, len(validateList)):
        f2.write(str(validateList[k].index) + " : " + str(validateList[k].satellite))
        f2.write("\n")

    f2 = open("runtimedata/testListNumbers.txt", "a")
    for k in range(0, len(testList)):
        f2.write(str(testList[k].index) + " : " + str(testList[k].satellite))
        f2.write("\n")

    for e in range(0, 20):
        random.shuffle(indexList)
        os.makedirs("C:/Users/Rehaan/Desktop/UruguayData/runtimedata/epoch" + str(e+19))
        for i in range (0, len(indexList)):
            #folder = random.randint(0, 8)
            imageSatCurrent = indexList[i]
            folder = 0
            # (i+1) is the length of our time series data
            print("testing example: -----------------------------------------" + str(i+1))
            print(folder)
            print("LEARNING RATE: " + str(learning_rate))
            f2 = open("runtimedata/learning_rate.txt", "a")
            f2.write(str(learning_rate) + "\n")
            currentIndex = imageSatCurrent.index
            if "SAME" == "SAME":
                if currentIndex + learning_window < len(satellite_images[folder]):
                    input = satellite_images[folder][currentIndex:(currentIndex+learning_window)]
                    correct_output = satellite_images[folder][currentIndex+learning_window]

                    first = False
                    if i == 0:
                        first = True

                    bptt(input, correct_output, 350*e + i, learning_window, currentIndex)

            if i%50 == 0 or i == len(indexList) - 1:
                print("-------------------Weight Matrix----------------")
                np.save('runtimedata/epoch' + str(e) + '/5main_kernelfinal3', cp.asnumpy(main_kernel))
                print("------------------connected_weights---------------------")
                np.save('runtimedata/epoch' + str(e) + '/5connected_weightsfinal3', cp.asnumpy(connected_weights))
                print("-------------------bias_y-------------------------")
                np.save('runtimedata/epoch' + str(e) + '/5bias_yfinal3', cp.asnumpy(bias_y))
                print("----------------------bias_o-----------------------")
                np.save('runtimedata/epoch' + str(e) + '/5bias_ofinal3', cp.asnumpy(bias_o))
                print("-------------------bias_c-------------------------")
                np.save('runtimedata/epoch' + str(e) + '/5bias_cfinal3', cp.asnumpy(bias_c))
                print("----------------------bias_f------------------")
                np.save('runtimedata/epoch' + str(e) + '/5bias_ffinal3', cp.asnumpy(bias_f))
                print("-----------------------bias_i-------------------")
                np.save('runtimedata/epoch' + str(e) + '/5bias_ifinal3', cp.asnumpy(bias_i))

        validate(validateList, e)
    test(validateList)

    

def produceRandomImageArray():
    global usableData
    list = []
    print("got here 2")
    for i in range(0, TOTAL_DATA - learning_window):
        list.append(make_ImageSat(i, "SAME"))
        print("----------------------adding-----------------------------")
        usableData += 1

    random.shuffle(list)        
    return list

def test(testList):
    global connected_weights
    global main_kernel
    global bias_i
    global bias_f
    global bias_c
    global bias_o
    global bias_y

    connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
    main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
    bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
    bias_o = cp.asarray(np.load('5bias_ofinal3.npy'))
    bias_c = cp.asarray(np.load('5bias_cfinal3.npy'))
    bias_f = cp.asarray(np.load('5bias_ffinal3.npy'))
    bias_i = cp.asarray(np.load('5bias_ifinal3.npy'))


    sumSquareError = np.zeros([M,N])

    for i in range (0, len(testList)):
        #folder = random.randint(0, 8)
        imageSatCurrent = testList[i]
        folder = 0
        currentIndex = imageSatCurrent.index
        print("---------------------WHAT-----------------------")
        print(str(currentIndex))
        if imageSatCurrent.satellite == "SAME":
            if currentIndex + learning_window + 2 < len(satellite_images[folder]):
                input = satellite_images[folder][currentIndex:(currentIndex+learning_window)]

                correct_output = satellite_images[folder][currentIndex+learning_window]
                
                roundArr = return_forecast(input, learning_window, currentIndex)

                true_prediction = unnormalize_np(correct_output[0], ndviMean, ndviStdDev)
                actual_prediction = unnormalize_np(roundArr, ndviMean, ndviStdDev)
                
                true_prediction = true_prediction*cloud_masks[currentIndex + learning_window]
                actual_prediction = actual_prediction*cloud_masks[currentIndex + learning_window]

                print("RMSE")
                print(rootmeansquare(true_prediction, actual_prediction))

                f2 = open("runtimedata/testResults.txt", "a")
                f2.write(str(rootmeansquare(true_prediction, actual_prediction)))
                f2.write("\n")

                sumSquareError = sumSquareError + (true_prediction - actual_prediction)**2

    
    sumSquareError = np.sqrt(sumSquareError/len(testList))
    finalValue = np.sum(sumSquareError)/10000
    print(str(finalValue))
    print(str(np.min(sumSquareError)))
    print(str(np.max(sumSquareError)))


def validate(validateList, e):
    global connected_weights
    global main_kernel
    global bias_i
    global bias_f
    global bias_c
    global bias_o
    global bias_y

    connected_weights = cp.asarray(np.load('runtimedata/epoch' + str(e) + '/5connected_weightsfinal3.npy'))
    main_kernel = cp.asarray(np.load('runtimedata/epoch' + str(e) + '/5main_kernelfinal3.npy'))
    bias_y = cp.asarray(np.load('runtimedata/epoch' + str(e) + '/5bias_yfinal3.npy'))
    bias_o = cp.asarray(np.load('runtimedata/epoch' + str(e) + '/5bias_ofinal3.npy'))
    bias_c = cp.asarray(np.load('runtimedata/epoch' + str(e) + '/5bias_cfinal3.npy'))
    bias_f = cp.asarray(np.load('runtimedata/epoch' + str(e) + '/5bias_ffinal3.npy'))
    bias_i = cp.asarray(np.load('runtimedata/epoch' + str(e) + '/5bias_ifinal3.npy'))

    global learning_rate
    global prev_validate

    average = 0

    sumSquareError = np.zeros([M,N])
    for i in range (0, len(validateList)):
        #folder = random.randint(0, 8)
        imageSatCurrent = validateList[i]
        folder = 0
        currentIndex = imageSatCurrent.index
        if imageSatCurrent.satellite == "SAME":
            if currentIndex + learning_window < len(satellite_images[folder]):
                input = satellite_images[folder][currentIndex:(currentIndex+learning_window)]

                correct_output = satellite_images[folder][currentIndex+learning_window]
                print(str(np.max(correct_output[0])) + " max NDVI")
                print(str(np.min(correct_output[1])) + " max rain")

                roundArr = return_forecast(input, learning_window, currentIndex)
            
                true_prediction = unnormalize_np(correct_output[0], ndviMean, ndviStdDev)
                actual_prediction = unnormalize_np(roundArr, ndviMean, ndviStdDev)

                true_prediction = true_prediction*cloud_masks[currentIndex + learning_window]
                actual_prediction = actual_prediction*cloud_masks[currentIndex + learning_window]
                
                f2 = open("runtimedata/validate1.txt", "a")
                f2.write(str(rootmeansquare(true_prediction, actual_prediction)))
                f2.write("\n")


                print("ROOT MEAN SQUARE: ")
                print(rootmeansquare(true_prediction, actual_prediction))

                # if rootmeansquare(true_prediction, actual_prediction) < 0.06:
                #     np.save("goodimage", true_prediction)
                #     np.save("predictimage", actual_prediction)

                average += rootmeansquare(true_prediction, actual_prediction)

                sumSquareError = sumSquareError + (true_prediction - actual_prediction)**2


    average = average/137
    if average>prev_validate:
        learning_rate = learning_rate * 0.8

    prev_validate = average
    sumSquareError = np.sqrt(sumSquareError/len(validateList))
    finalValue = np.sum(sumSquareError)/10000
    f2 = open("runtimedata/validate2.txt", "a")
    f2.write(str(finalValue) + "\n")
    f2.write(str(np.min(sumSquareError)) + "\n")
    f2.write(str(np.max(sumSquareError)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("runtimedata/validate1.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

loadData()
