# A simple ConvRNN, used a baseline to compare the convLSTM, as well 
# as the "AnnualGate" convLSTM to.

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

print(random.randint(1, 19))
np.set_printoptions(threshold=np.inf)

# number of sets of images
S = 10

# number of images per sequence/set
T = 711

# dimensions of the image
M = 100
N = 100

channels_img = 2  # antecedent NDVI and rain
channels_hidden = 8
kernel_dimension = 5
pad_constant = 2

satellite_images = np.empty([S, 711, channels_img, M, N])
learning_window = 10

prev_validate = 100

clip_threshold = 1
clip_threshold_output = 1

r_kernel_tanh = math.sqrt(6/((channels_hidden+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden))
r_kernel_sigmoid = math.sqrt(6/((channels_hidden+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden))
r_connected_weights = math.sqrt(6/(channels_hidden + 1))

main_kernel = cp.random.uniform(-r_kernel_tanh, r_kernel_tanh, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
connected_weights = cp.random.normal(-r_connected_weights, r_connected_weights, (1, channels_hidden))

bias_h = cp.zeros([channels_hidden, M, N])
bias_y = cp.zeros([channels_img, M, N])

learning_rate = 0.0003
learning_rate_counter = 0

ndviMean = 0.66673688145266
ndviStdDev = 0.16560766237944935
rainMean = 0.19636724781555773
rainStdDev = 0.16560766237944935
net_loss = 0

class ImageSat(object):
    index = 0
    satellite = "SAME"

    def __init__(self, index, satellite):
        self.index = index
        self.satellite = satellite

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

def rect_linear(arr):
    newArr = copy.deepcopy(arr)
    newArr[arr<0] = 0
    return newArr

def rect_linear_derivative(arr):
    newArr = cp.zeros(arr.shape)
    newArr[arr>0] = 1
    return newArr

def make_ImageSat(index, satellite):
    imageSat = ImageSat(index, satellite)
    return imageSat

# x[t] is the input
def forward_prop(x,local_time):
    global learning_rate
    # Hidden Unit
    h = cp.empty([local_time + 1, channels_hidden, M, N])
    h[-1] = cp.zeros([channels_hidden, M, N])

    for t in np.arange(local_time):
        temporary = cp.concatenate((x[t], h[t - 1]), axis=0)
        temporary = temporary.reshape(1, channels_img + channels_hidden, M, N)

        h[t] = tanh(cp.asarray(F.convolution_2d(temporary, main_kernel, b=None, pad=pad_constant)[0].data) + bias_h)
                                      
    output = cp.asarray(F.convolution_2d(h[local_time-1].reshape(1, channels_hidden, M, N), connected_weights.reshape(1, channels_hidden, 1, 1), b = None, pad = 0)[0][0].data) + bias_y[0]
    true_output = sigmoid(output)  
  
    return true_output, output, cp.reshape(h[local_time-1], (channels_hidden, M*N)), h


def calculate_loss2(prediction, y):
    return -np.sum(np.multiply(y, np.log(prediction)) + np.multiply(np.ones(y.shape) - y, np.log(np.ones(y.shape) - prediction)))

#root mean square error
def rootmeansquare(prediction, y):
    return cp.sqrt(cp.sum((prediction - y)**2)/(10000))

# Calculate loss
def calculate_loss(prediction, y):
    lossExpression = 0.5*cp.sum((prediction - y)**2)
    return lossExpression

def calculate_loss_modified(prediction, y):
    prediction[prediction == 0] = 0.00000001
    y[y == 0] = 0.00000001
    lossExpression = -cp.sum(cp.multiply(y, cp.log(prediction)) + cp.multiply(cp.ones(y.shape) - y, cp.log(cp.ones(y.shape) - prediction)))
    return lossExpression

def return_forecast(x, local_time):
    a,b,c,d = forward_prop(cp.asarray(x), local_time)
    return cp.asnumpy(a)

def loss_derivative(x, y):
    return (x-y)

def bptt(x2, y2, iteration, local_time):
    x = cp.asarray(x2)
    y = cp.asarray(y2)

    global connected_weights
    global main_kernel
    global bias_y
    global bias_h

    global learning_rate
    global learning_rate_counter

    # Perform forward prop
    prediction, pre_sigmoid_prediction, hidden_prediction, h = forward_prop(x, local_time)

    loss = calculate_loss(prediction, y[0])
    print("LOSS BEFORE: ")
    print(loss)

    lossExact = calculate_loss_modified(y[0], y[0])
    
    # Calculate loss with respect to final layer
    dLdy_2 = loss_derivative(prediction, y[0])
    # Calculate loss with respect to pre sigmoid layer
    dLdy_1 = cp.multiply(sigmoid_derivative(pre_sigmoid_prediction), dLdy_2)
    
    # Calculate loss with respect to last layer of lstm
    dLdh = cp.asarray(F.convolution_2d(dLdy_1.reshape(1, 1, M, N), (connected_weights.reshape(1, channels_hidden, 1, 1)).transpose(1,0,2,3), b=None, pad=0)[0].data)
    dLdw_0 = cp.asarray(F.convolution_2d(hidden_prediction.reshape(channels_hidden, 1, M, N), dLdy_1.reshape(1, 1, M, N), b=None, pad=0).data).transpose(1,0,2,3)
    dLdb_y = dLdy_1
    dLdw_0 = dLdw_0.reshape(1, channels_hidden)

    # uncomment if gradient clipping output is necessary
    # if cp.linalg.norm(dLdw_0) > clip_threshold_output:
    #    dLdw_0 = dLdw_0*clip_threshold_output/cp.linalg.norm(dLdw_0)
    # if cp.linalg.norm(dLdb_y) > clip_threshold_output:
    #    dLdb_y = dLdb_y*clip_threshold_output/cp.linalg.norm(dLdb_y)

    #--------------------fully connected------------------
    bias_y = bias_y - learning_rate*dLdb_y
    connected_weights = connected_weights - learning_rate*dLdw_0

    dLdW = cp.zeros([channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension])

    # initialize biases
    dLdb_h = cp.zeros([channels_hidden, M, N])

    for t in cp.arange(local_time - 1, -1, -1):
        dLdh = cp.multiply(dLdh, (cp.ones((channels_hidden, M, N)) - cp.multiply(h[t], h[t]))) #dLdh_hat

        temporary = cp.concatenate((x[t], h[t - 1]), axis=0).reshape(channels_hidden + channels_img, 1, M, N)

        dLdI = cp.asarray(F.convolution_2d(dLdh.reshape(1, channels_hidden, M, N), main_kernel.transpose(1, 0, 2, 3), b=None, pad=pad_constant)[0].data) # reshape into flipped kernel dimensions
        dLdW_temp = cp.asarray((F.convolution_2d(temporary, dLdh.reshape(channels_hidden, 1, M, N), b=None, pad=pad_constant).data).transpose(1,0,2,3)) #reshape into kernel dimensions

        # clip the derivatives before modifying the weights and biases
        if cp.linalg.norm(dLdW_temp) > clip_threshold:
            dLdW_temp = dLdW_temp*clip_threshold/cp.linalg.norm(dLdW_temp)
        if cp.linalg.norm(dLdh) > clip_threshold:
            dLdh = dLdh*clip_threshold/cp.linalg.norm(dLdh)

        dLdW += dLdW_temp
        dLdb_h += dLdh

        # Reinitialize
        dLdh = dLdI[channels_img: channels_img+channels_hidden]

    # gradient clipping step
    if cp.linalg.norm(dLdW_temp) > clip_threshold:
        dLdW_temp = dLdW_temp*clip_threshold/cp.linalg.norm(dLdW_temp)
    if cp.linalg.norm(dLdh) > clip_threshold:
        dLdh = dLdh*clip_threshold/cp.linalg.norm(dLdh)

    #---------------------update main kernel---------
    main_kernel = main_kernel - learning_rate*dLdW
    #--------------------update bias h-----------------------
    bias_h = bias_h - learning_rate*dLdb_h

    prediction2 = forward_prop(x, local_time)[0];

    loss2 = calculate_loss(prediction2, y[0])


    rms3 = rootmeansquare(unnormalize_cp(prediction2, ndviMean, ndviStdDev), unnormalize_cp(y[0], ndviMean, ndviStdDev))
    print("LOSS AFTER: ")
    print(loss2)
    
    f2 = open("loss.txt", "a")
    f2.write(str(rms3) + "\n")

def loadData():
    global satellite_images

    satellite_images = np.empty([S, 711, channels_img, M, N])

    dataNDVI = np.zeros([14])
    dataRain = np.zeros([10])

    #totalnumber is 771
    ndvi_images = np.zeros([711, M, N])
    rain_images = np.zeros([711, M, N])

    totalNumber = 0
    outer = 0
    for a in range(0,1):
        for b in range(0, 13):
            counter = 0
            if os.path.isdir("combined_images/combine_" + str(a) + "_" + str(b)):
                filenames = []
                for filename in os.listdir("combined_images/combine_" + str(a) + "_" + str(b)): 
                    if filename[len(filename)-6:len(filename)-4] == "1d":
                        filenames.append(filename)
                filenames.sort()
                for filename in filenames:
                    arr = np.load("combined_images/combine_" + str(a) + "_" + str(b) + "/" + filename)
                    if counter<len(satellite_images[outer]):
                        satellite_images[outer][counter][0] = normalize_np(arr[0], ndviMean, ndviStdDev)
                        satellite_images[outer][counter][1] = arr[1]/0.7#(arr[1] - 0.35)*(0.5/0.35) #normalize_np(arr[1], rainMean, rainStdDev)
                        print(totalNumber)
                        ndvi_images[totalNumber] = normalize_np(arr[0], ndviMean, ndviStdDev)
                        rain_images[totalNumber] = arr[1]/0.7 #normalize_np(arr[1], rainMean, rainStdDev)
                    counter +=1
                    totalNumber += 1
                print(str(a) + " : " + str(b))
                
    print("AQUA Data Loaded")

    list1 = produceRandomImageArray()

    main(list1)

def MAPE(correct, prediction):
    return np.sum(np.absolute(correct-prediction)/correct)/100
    correct[correct == 0] = 0.000001
    prediction[prediction == 0] = 0.000001

def main(indexGeneralList):
    #initiate training process etc
    global stdev
    global mean
    global learning_rate

    if LOAD_PREV_WEIGHTS == True:
        global connected_weights
        global main_kernel
        global bias_h
        global bias_y

        connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
        main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
        bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
        bias_h = cp.asarray(np.load('5bias_hfinal3.npy'))

    maxNdvi = 0
    maxRain = 0

    for i in range(0, T-1):
       correct_output = satellite_images[0][i]
       maxN = np.max(correct_output[0])
       maxR = np.max(correct_output[1])
       if maxN > maxNdvi: 
           maxNdvi = maxN
       if maxR > maxRain:
           maxRain = maxR
    print("maxNdvi: " + str(maxNdvi))
    print("maxRain: " + str(maxRain))

    indexList = indexGeneralList[0:480] 
    validateList = indexGeneralList[480:617]
    testList = indexGeneralList[617:686]
    
    f2 = open("indexListNumbers.txt", "a")
    for k in range(0, 480):
        f2.write(str(indexList[k].index) + " : " + str(indexList[k].satellite))
        f2.write("\n")

    f2 = open("validateListNumbers.txt", "a")
    for k in range(0, 137):
        f2.write(str(validateList[k].index) + " : " + str(validateList[k].satellite))
        f2.write("\n")

    f2 = open("testListNumbers.txt", "a")
    for k in range(0, 69):
        f2.write(str(testList[k].index) + " : " + str(testList[k].satellite))
        f2.write("\n")

    for e in range(0, 50):
        random.shuffle(indexList)
        for i in range (0, len(indexList)):
            imageSatCurrent = indexList[i]
            folder = 0
            # (i+1) is the length of our time series data
            print("testing example: -----------------------------------------" + str(i+1))
            print(folder)
            
            print("LEARNING RATE: " + str(learning_rate))
            currentIndex = imageSatCurrent.index
            if currentIndex + learning_window < len(satellite_images[folder]):
                input = satellite_images[folder][currentIndex:(currentIndex+learning_window)]

                correct_output = satellite_images[folder][currentIndex+learning_window]

                print(str(np.max(correct_output[0])) + " max NDVI")
                print(str(np.max(correct_output[1])) + " max rain")
                print(str(np.min(correct_output[0])) + " min NDVI")
                print(str(np.min(correct_output[1])) + " min rain")

                bptt(input, correct_output, 350*e + i, learning_window)

            if i%50 == 0:
                print("-------------------Weight Matrix----------------")
                np.save('5main_kernelfinal3', cp.asnumpy(main_kernel))
                print("------------------connected_weights---------------------")
                np.save('5connected_weightsfinal3', cp.asnumpy(connected_weights))
                print("-------------------bias_y-------------------------")
                np.save('5bias_yfinal3', cp.asnumpy(bias_y))
                print("----------------------bias_o-----------------------")
                np.save('5bias_hfinal3', cp.asnumpy(bias_h))

        print("-------------------Weight Matrix----------------")
        np.save('5main_kernelfinal3', cp.asnumpy(main_kernel))
        print("------------------connected_weights---------------------")
        np.save('5connected_weightsfinal3', cp.asnumpy(connected_weights))
        print("-------------------bias_y-------------------------")
        np.save('5bias_yfinal3', cp.asnumpy(bias_y))
        print("----------------------bias_o-----------------------")
        np.save('5bias_hfinal3', cp.asnumpy(bias_h))
        validate(validateList)
    
    test(testList)

def produceRandomImageArray():
    list = []
    for i in range(0, 686):
        list.append(make_ImageSat(i, "SAME"))

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
    bias_h = cp.asarray(np.load('5bias_hfinal3.npy'))

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
                print(str(np.max(correct_output[0])) + " max NDVI")
                print(str(np.min(correct_output[1])) + " max rain")

                print(correct_output[0][0])

                roundArr = return_forecast(input, learning_window)

                true_prediction = unnormalize_np(correct_output[0], ndviMean, ndviStdDev)
                actual_prediction = unnormalize_np(roundArr, ndviMean, ndviStdDev)

                f2 = open("testResults.txt", "a")
                f2.write(str(rootmeansquare(true_prediction, actual_prediction)))
                f2.write("\n")

                sumSquareError = sumSquareError + (true_prediction - actual_prediction)**2

    
    sumSquareError = np.sqrt(sumSquareError/len(testList))
    finalValue = np.sum(sumSquareError)/10000
    print(str(finalValue))
    print(str(np.min(sumSquareError)))
    print(str(np.max(sumSquareError)))


def validate(validateList):
    global connected_weights
    global main_kernel
    global bias_h
    global bias_y

    connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
    main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
    bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
    bias_h = cp.asarray(np.load('5bias_hfinal3.npy'))

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
                longago = satellite_images[folder][currentIndex - 24]
                #print(str(np.max(correct_output[0])) + " max NDVI")
                #print(str(np.min(correct_output[1])) + " max rain")

                roundArr = return_forecast(input, learning_window)
            
                true_prediction = unnormalize_np(correct_output[0], ndviMean, ndviStdDev)
                actual_prediction = unnormalize_np(roundArr, ndviMean, ndviStdDev)
                longagoactual = unnormalize_np(longago[0], ndviMean, ndviStdDev)

                f2 = open("validate1.txt", "a")
                f2.write(str(rootmeansquare(true_prediction, actual_prediction)))
                f2.write("\n")

                # print(rootmeansquare(true_prediction, actual_prediction))

                # if rootmeansquare(true_prediction, actual_prediction) > 0.2:
                #     print(currentIndex)
 
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
    f2 = open("validate2.txt", "a")
    f2.write(str(finalValue) + "\n")
    f2.write(str(np.min(sumSquareError)) + "\n")
    f2.write(str(np.max(sumSquareError)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate1.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

def average_loss():
    global connected_weights
    global main_kernel
    global bias_i
    global bias_f
    global bias_c
    global bias_o
    global bias_y
    connected_weights = cp.asarray(np.load('5connected_weightsr.npy'))
    main_kernel = cp.asarray(np.load('5main_kernelr.npy'))
    bias_y = cp.asarray(np.load('5bias_yr.npy'))
    bias_o = cp.asarray(np.load('5bias_or.npy'))
    bias_c = cp.asarray(np.load('5bias_cr.npy'))
    bias_f = cp.asarray(np.load('5bias_fr.npy'))
    bias_i = cp.asarray(np.load('5bias_ir.npy'))

    data = np.load('mnist_test_seq.npy')
    data = data.transpose(1, 0, 2, 3)
    realdata = data/255.0
    sum = 0.0
    for i in range(2000, 10000):
        input = realdata[i][0:10]
        print("Current loss for: " +str(i))
        print(calculate_loss2(return_forecast(input.reshape(10, 1, 64, 64)), realdata[i][10]))

loadData()
