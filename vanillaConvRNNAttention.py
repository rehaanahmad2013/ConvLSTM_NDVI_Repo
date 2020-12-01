# A vanilla ConvRNN model, but fitted with an attention network that allows
# it to focus on previous year's NDVI -- this allows the model to exploit the
# annual periodic trends present in the NDVI.

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

learning_rate = 0.001

# number of sets of images
S = 1

# number of images per sequence/set
T = 410

tau_len = 15
LOAD_PREV_WEIGHTS = False

# dimensions of the image
M = 100
N = 100

# how far to look back at x tau
distance = 37
distance_forward = 10
learning_window = 15

channels_img = 2  # antecedent NDVI and rain
channels_hidden = 8
kernel_dimension = 5
kernel_dimension_g = 5
kernel_dimension_p = 5
channels_p = 1
pad_constant = 2

clip_threshold = 1
attention_clip_threshold = 10

#this values are saved for producing the final output that must be displayed to the user
stdev = 0
mean = 0

satellite_images = np.empty([S, 711, channels_img, M, N])

#---------------------------------------He Normal Initialization-------------------------------------
r_v_connected_weights = 2*math.sqrt(6/(channels_hidden*M*N + 1))
r_e_kernel = 2*math.sqrt(6/(channels_hidden + (channels_img + channels_hidden)*(kernel_dimension)*(kernel_dimension)))

v_connected_weights = cp.random.uniform(-r_v_connected_weights, r_v_connected_weights,(channels_hidden*M*N))
e_kernel = cp.random.uniform(-r_e_kernel, r_e_kernel, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))


r_kernel_tanh = 0.5*math.sqrt(6/((channels_hidden+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden))
r_kernel_sigmoid = math.sqrt(6/((channels_hidden+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden))
r_connected_weights =  1.2*math.sqrt(6/(channels_hidden + 1))
main_kernel = cp.random.uniform(-r_kernel_tanh, r_kernel_tanh, (channels_hidden, channels_p + channels_img + channels_hidden, kernel_dimension, kernel_dimension))
connected_weights = cp.random.normal(-r_connected_weights, r_connected_weights, (1, channels_hidden))

ndviMean = 0.66673688145266
ndviStdDev = 0.16560766237944935
rainMean = 0.19636724781555773
rainStdDev = 0.16560766237944935

prev_validate = 100
learning_rate = 0.000192
net_loss = 0
learning_rate_counter = 0

bias_h = cp.zeros([channels_hidden, M, N])
bias_y = cp.zeros([channels_img, M, N])
bias_e =  cp.zeros([channels_hidden, M, N])
bias_v = cp.zeros([distance_forward])

class ImageSat(object):
    index = 0
    satellite = "SAME"

    def __init__(self, index, satellite):
        self.index = index
        self.satellite = satellite

def make_ImageSat(index, satellite):
    imageSat = ImageSat(index, satellite)
    return imageSat

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = cp.exp(x - cp.max(x))
    return e_x / cp.sum(e_x)

def softmax_derivative(x):
    return softmax(x)*(1 - softmax(x))

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

def expdecay(x):
    return distance/(1+cp.exp(-0.01*x))

def rect_linear_exponential(arr):
    arr2 = copy.deepcopy(arr)
    arr2 = expdecay(arr2)
    return arr2

def rect_linear_exponential_derivative(arr):
    arr2 = copy.deepcopy(arr)
    derivatives = 0.01*115*cp.exp(-0.01*arr2)/((1+cp.exp(-0.01*arr2))**2)
    return derivatives

def rect_linear(arr):
    newArr = copy.deepcopy(arr)
    newArr[arr<0] = 0
    return newArr

def rect_linear_derivative(arr):
    newArr = cp.zeros(arr.shape)
    newArr[arr>0] = 1
    return newArr

def normalize_np(arr, mean, stddev):
    return (arr + 0.3)/1.3
    #return 1/(1+np.exp(-(arr-mean)/stddev))

def unnormalize_np(arr, mean, stddev):
    return arr*1.3-0.3
    #return mean - stddev*np.log((1-arr)/arr)

def normalize_cp(arr, mean, stddev):
    return (arr + 0.3)/1.3
    #return 1/(1+cp.exp(-(arr-mean)/stddev))

def unnormalize_cp(arr, mean, stddev):
    return arr*1.3-0.3
    #return mean - stddev*cp.log((1-arr)/arr)

# x[t] is the input
def forward_prop(x, local_time, sequence, isFirst, timestamp, satellite_name):

    s = cp.empty([local_time, distance_forward, channels_hidden, M, N])

    e = cp.empty([local_time, distance_forward])

    alpha = cp.empty([local_time, distance_forward])

    p = cp.empty([local_time, channels_p, M, N])

    # Hidden Unit
    h = cp.empty([local_time + 1, channels_hidden, M, N])
    h[-1] = cp.zeros([channels_hidden, M, N])
    # LSTM FORWARD PROPAGATION
    for t in np.arange(local_time):

        # Attention Network
        for z in range(timestamp + t - (distance + learning_window), timestamp + distance_forward + t - (distance + learning_window)):
            temp = cp.concatenate((cp.asarray(satellite_images[sequence][z]), h[t - 1]), axis = 0)
            s[t][z - (timestamp + t - (distance + learning_window))] = tanh(cp.asarray(F.convolution_2d(temp.reshape(1, channels_img + channels_hidden, M, N), e_kernel, b=None, pad=pad_constant)[0].data) + bias_e)
            s_temp = s[t][z - (timestamp + t - (distance + learning_window))].reshape(M*N*channels_hidden)
            e[t][z - (timestamp + t - (distance + learning_window))] = cp.dot(v_connected_weights, s_temp) + bias_v[z - (timestamp + t - (distance + learning_window))]

        xtemp = satellite_images[sequence][timestamp - distance:timestamp-distance+distance_forward, 0]

        alpha[t] = softmax(e[t])
        p[t] = cp.tensordot(alpha[t], cp.asarray(xtemp), axes = 1).reshape(1, M, N) # Sum all x arrays up, weighted array

        temporary = cp.concatenate((x[t],  p[t], h[t-1]), axis = 0)
        temporary = temporary.reshape(1, channels_img + channels_p + channels_hidden, M, N)

        h[t] = tanh(cp.asarray(F.convolution_2d(temporary, main_kernel, b=None, pad=2)[0].data) + bias_h)


    # 1 x 1 convolution
    output = cp.matmul(connected_weights, h[local_time-1].reshape(channels_hidden, M * N)).reshape(M, N) + bias_y[0]
    true_output = rect_linear(output)
    
    return true_output, output, cp.reshape(h[local_time-1], (channels_hidden, M*N)), p, h, s, e, alpha, xtemp


def calculate_loss2(prediction, y):
  #  prediction[prediction<0.1] = 0.00000001
    return -np.sum(np.multiply(y, np.log(prediction)) + np.multiply(np.ones(y.shape) - y, np.log(np.ones(y.shape) - prediction)))

#root mean square error
def rootmeansquare(prediction, y):
    return cp.sqrt(cp.sum((prediction - y)**2)/(10000)) 

# Calculate loss
def calculate_loss(prediction, y):
    lossExpression = 0.5*cp.sum((prediction - y)**2)
    return lossExpression

# Calculate loss
def calculate_loss_modified(prediction, y):
    prediction[prediction == 0] = 0.00000001
    y[y == 0] = 0.00000001
    lossExpression = -cp.sum(cp.multiply(y, cp.log(prediction)) + cp.multiply(cp.ones(y.shape) - y, cp.log(cp.ones(y.shape) - prediction)))
    return lossExpression

def return_forecast(x, learning_window, region, timestamp, satellite_name):
    prediction, pre_sigmoid_prediction, hidden_prediction, p, h, s, e, alpha, X_array = forward_prop(cp.asarray(x), learning_window, region, False, timestamp, satellite_name)
    return cp.asnumpy(prediction), cp.asnumpy(p)

def loss_derivative(x, y):
    return (x-y)
    
def bptt(x2, y2, iteration, local_time, region, isFirst, timestamp, satellite_name):
    x = cp.asarray(x2)
    y = cp.asarray(y2)

    global connected_weights
    global main_kernel
    global bias_y
    global e_kernel
    global learning_rate
    global v_connected_weights
    global bias_h
    global bias_e
    global bias_v

    # Perform forward prop

    global net_loss
    global learning_rate
    global learning_rate_counter

    #CHANGE
    prediction, pre_sigmoid_prediction, hidden_prediction, p, h, s, e, alpha, xtemp = forward_prop(x, local_time, 0, False, timestamp, "SAME")

    #Any NDVI with 0 is water, and will remain water. NDVI is only applicable to vegatation, thus just make the prediction 0 at every point the previous NDVI is 0.
    loss = calculate_loss(prediction, y[0])
    print("LOSS BEFORE: ")
    print(loss)
    # Calculate loss with respect to final layer
    dLdy_2 = loss_derivative(prediction, y[0])
    # Calculate loss with respect to pre sigmoid layer
    dLdy_1 = cp.multiply(rect_linear_derivative(pre_sigmoid_prediction), dLdy_2)
    # Calculate loss with respect to last layer of lstm
    testArr = cp.reshape(cp.matmul(cp.transpose(connected_weights), dLdy_1.reshape(1, M * N)), (channels_hidden, M, N))
    dLdh =  testArr # initial value of dLdh

    dLdw_0 = cp.matmul(dLdy_1.reshape(1, M*N), hidden_prediction.transpose(1,0))

    # Calculate loss with respect to bias y
    dLdb_y = dLdy_1
    #--------------------fully connected------------------
    bias_y = bias_y - learning_rate*dLdb_y
    connected_weights = connected_weights - learning_rate*dLdw_0

    # Initialize weight matrices
    dLdW = cp.zeros([channels_hidden, channels_p + channels_img + channels_hidden, kernel_dimension, kernel_dimension])
    dLdW_v = cp.zeros([channels_hidden*M*N])
    dLdW_e = cp.zeros([channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension])
    
    # initialize biases
    dLdb_e = cp.zeros([channels_hidden, M, N])
    dLdb_h = cp.zeros([channels_hidden, M, N])
    dLdb_v = cp.zeros([distance_forward])

    for t in cp.arange(local_time - 1, -1, -1):
        dLdh = cp.multiply(dLdh, (cp.ones((channels_hidden, M, N)) - cp.multiply(h[t], h[t]))) #dLdh_hat

        temporary = cp.concatenate((x[t], p[t], h[t - 1]), axis=0).reshape(channels_hidden + channels_img + channels_p, 1, M, N)

        dLdI = cp.asarray(F.convolution_2d(dLdh.reshape(1, channels_hidden, M, N), main_kernel.transpose(1, 0, 2, 3), b=None, pad=2)[0].data) # reshape into flipped kernel dimensions
        dLdW_temp = cp.asarray((F.convolution_2d(temporary, dLdh.reshape(channels_hidden, 1, M, N), b=None, pad=2).data).transpose(1,0,2,3)) #reshape into kernel dimensions

        #create dLdp, which is the derivative of loss with respect to p
        dLdp = dLdI[channels_img: channels_img + channels_p]
        #------------------------------------------ATTENTION BACKPROPAGATION CODE-------------------------------------
        dLdAlpha = cp.zeros(distance_forward)
        for k in range(0, distance_forward):
            dLdAlpha[k] = cp.sum(dLdp*cp.asarray(xtemp[k]))

        dLde = dLdAlpha*softmax_derivative(e[t])
        dLdW_v_temp = cp.zeros([channels_hidden*M*N])
        dLdW_e_temp = cp.zeros([channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension])
        dLdh_temp = cp.zeros([channels_hidden, M, N])
        for k in range(0, distance_forward):
            dLdW_v_temp += dLde[k]*s[t][k].reshape((M*N*channels_hidden))
            dLds = dLde[k]*v_connected_weights # Changes each iteration of nested loop
            dLds = dLds.reshape((channels_hidden, M, N))
            dLds = cp.multiply(dLds, (cp.ones((channels_hidden, M, N)) - cp.multiply(s[t][k], s[t][k])))
            temp3 = cp.concatenate((cp.asarray(satellite_images[region][k + timestamp - distance]), h[t - 1]), axis = 0)


            dLdI_e = cp.asarray(F.convolution_2d(dLds.reshape(1, channels_hidden, M, N), e_kernel.transpose(1, 0, 2, 3), b=None, pad=pad_constant)[0].data) # reshape into flipped kernel dimensions
            dLdW_e_temp += cp.asarray((F.convolution_2d(temp3.reshape(channels_hidden+channels_img, 1, M, N), dLds.reshape(channels_hidden, 1, M, N), b=None, pad=pad_constant).data).transpose(1,0,2,3)) #reshape into kernel dimensions
            dLdh_temp += dLdI_e[channels_img: channels_img + channels_hidden]
            if cp.amax(dLds) > 1 or cp.amin(dLds) < -1:
                dLds = dLds/cp.linalg.norm(dLds)
            dLdb_e += dLds

        #---------------------------------------------UPDATE DERIVATIVES-------------------------------------
        dLdW += dLdW_temp
        dLdb_h += dLdh
        dLdb_v += dLde.reshape([distance_forward])
        # Reinitialize
        dLdh = dLdI[channels_img + channels_p: channels_img + channels_p + channels_hidden]


     #Clip all gradients again
    if cp.linalg.norm(dLdW) > clip_threshold:
        dLdW = dLdW*clip_threshold/cp.linalg.norm(dLdW)
    if cp.linalg.norm(dLdW_e) > clip_threshold:
        dLdW_e = dLdW_e*clip_threshold/cp.linalg.norm(dLdW_e)
    if cp.linalg.norm(dLdW_v) > clip_threshold:
        dLdW_v = dLdW_v*clip_threshold/cp.linalg.norm(dLdW_v)
    if cp.linalg.norm(dLdb_h) > clip_threshold:
        dLdb_h = dLdb_h*clip_threshold/cp.linalg.norm(dLdb_h)
    if cp.linalg.norm(dLdb_e) > clip_threshold:
        dLdb_e = dLdb_e*clip_threshold/cp.linalg.norm(dLdb_e)
    if cp.linalg.norm(dLdb_v) > clip_threshold:
        dLdb_v = dLdb_v*clip_threshold/cp.linalg.norm(dLdb_v)

    #---------------------------------------UPDATE WEIGHTS----------------------------------
    #---------------------update main kernel---------
    main_kernel = main_kernel - learning_rate*dLdW
    #---------------------update e kernel---------
    e_kernel = e_kernel - learning_rate*dLdW_e
    #---------------------update v_connected_weights---------
    v_connected_weights = v_connected_weights - learning_rate*dLdW_v
    #--------------------update bias h-----------------------
    bias_h = bias_h - learning_rate*dLdb_h
    #--------------------update bias e-----------------------
    bias_e = bias_e - learning_rate*dLdb_e
    #--------------------update bias v-----------------------
    bias_v = bias_v - learning_rate*dLdb_v

    prediction2, pre_sigmoid_prediction2, hidden_prediction2, p2, h2, s2, e2, alpha2, xtemp2 = forward_prop(x, local_time, 0, False, timestamp, "SAME")
    loss2 = calculate_loss(prediction2, y[0])
    print("LOSS AFTER: ")
    print(loss2)
    loss3 = calculate_loss(prediction2, y[0])
    rms3 = rootmeansquare(unnormalize_cp(prediction2, ndviMean, ndviStdDev), unnormalize_cp(y[0], ndviMean, ndviStdDev))
    print("LOSS AFTER WATER: ")
    print(loss3)
    f2 = open("loss.txt", "a")
    f2.write(str(rms3) + "\n")

    learning_rate_counter +=1
    net_loss += (loss2 - loss)
    if learning_rate_counter == 10:
        print("----------------------------NET LOSS OF 10 EXAMPLES-----------------------------")
        print(net_loss)
        learning_rate_counter = 0
        #if net_loss > 0:
            #learning_rate = learning_rate * 0.8
        net_loss = 0

    print("backpropagation complete")

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

    print(np.mean(ndvi_images))
    print(np.mean(rain_images))

    print(np.std(ndvi_images))
    print(np.std(rain_images))

    print(np.min(ndvi_images))
    print(np.min(rain_images))

    print(np.max(ndvi_images))
    print(np.max(rain_images))

    list1 = produceRandomImageArray()

    main(list1)

def MAPE(correct, prediction):
    return np.sum(np.absolute(correct-prediction)/correct)/100
    correct[correct == 0] = 0.000001
    prediction[prediction == 0] = 0.000001

def main(indexGeneralList):
    global connected_weights
    global main_kernel
    global e_kernel
    global v_connected_weights
    global bias_h
    global bias_y
    global bias_e
    global bias_v

    if LOAD_PREV_WEIGHTS == True:
        e_kernel = cp.asarray(np.load('5e_kernelfinal3.npy'))
        v_connected_weights = cp.asarray(np.load('5v_connected_weightsfinal3.npy'))
        bias_e = cp.asarray(np.load('5bias_efinal3.npy'))
        connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
        main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
        bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
        bias_h = cp.asarray(np.load('5bias_hfinal3.npy'))
        bias_v = cp.asarray(np.load('5bias_vfinal3.npy'))

    #initiate training process etc
    global stdev
    global mean
    global learning_rate

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
       print("ndvi: " + str(maxN))
       print("rain: " + str(maxR))
    print("maxNdvi: " + str(maxNdvi))
    print("maxRain: " + str(maxRain))

    indexList = indexGeneralList[0:442] 
    validateList = indexGeneralList[442:568]
    testList = indexGeneralList[568:631]
    
    # f2 = open("indexListNumbers.txt", "a")
    # for k in range(0, 442):
    #     f2.write(str(indexList[k].index) + " : " + str(indexList[k].satellite))
    #     f2.write("\n")

    # f2 = open("validateListNumbers.txt", "a")
    # for k in range(0, 126):
    #     f2.write(str(validateList[k].index) + " : " + str(validateList[k].satellite))
    #     f2.write("\n")

    # f2 = open("testListNumbers.txt", "a")
    # for k in range(0, 63):
    #     f2.write(str(testList[k].index) + " : " + str(testList[k].satellite))
    #     f2.write("\n")

    for e in range(0, 1):
        random.shuffle(indexList)
        for i in range (0, len(indexList)):
            #folder = random.randint(0, 8)
            imageSatCurrent = indexList[i]
            folder = 0
            # (i+1) is the length of our time series data
            print("testing example: -----------------------------------------" + str(i+1))
            print(folder)
            print("LEARNING RATE: " + str(learning_rate))
            currentIndex = imageSatCurrent.index
            if imageSatCurrent.satellite == "SAME":
                if currentIndex + learning_window < len(satellite_images[folder]):
                    input = satellite_images[folder][currentIndex:(currentIndex+learning_window)]

                    correct_output = satellite_images[folder][currentIndex+learning_window]

                    print(str(np.max(correct_output[0])) + " max NDVI")
                    print(str(np.min(correct_output[1])) + " max rain")

                    first = False
                    if i == 0:
                        first = True

                    bptt(input, correct_output, i, learning_window, folder, first, currentIndex, "SAME")

            if i%50 == 0:
                print("-------------------Weight Matrix----------------")
                np.save('5main_kernelfinal3', cp.asnumpy(main_kernel))
                print("------------------connected_weights---------------------")
                np.save('5connected_weightsfinal3', cp.asnumpy(connected_weights))
                print("-------------------e_kernel-------------------------")
                np.save('5e_kernelfinal3', cp.asnumpy(e_kernel))
                print("-------------------------bias_h--------------------")
                np.save('5bias_hfinal3', cp.asnumpy(bias_h))
                print("-------------------bias_y-------------------------")
                np.save('5bias_yfinal3', cp.asnumpy(bias_y))
                print("-----------------------bias_e-------------------")
                np.save('5bias_efinal3', cp.asnumpy(bias_e))
                print("-----------------------bias_v-------------------")
                np.save('5bias_vfinal3', cp.asnumpy(bias_v))
                print("-----------------------v_connected_weights-------------------")
                np.save('5v_connected_weightsfinal3', cp.asnumpy(v_connected_weights))

        validate(validateList)

    test(testList)

def produceRandomImageArray():
    list = []
    for i in range(55, 686):
        list.append(make_ImageSat(i, "SAME"))

    random.shuffle(list)        
    return list

def test(testList):
    global connected_weights
    global main_kernel
    global e_kernel
    global v_connected_weights
    global bias_h
    global bias_y
    global bias_e
    global bias_v

    e_kernel = cp.asarray(np.load('5e_kernelfinal3.npy'))
    v_connected_weights = cp.asarray(np.load('5v_connected_weightsfinal3.npy'))
    bias_e = cp.asarray(np.load('5bias_efinal3.npy'))
    connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
    main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
    bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
    bias_h = cp.asarray(np.load('5bias_hfinal3.npy'))
    bias_v = cp.asarray(np.load('5bias_vfinal3.npy'))

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

                roundArr, p = return_forecast(input, learning_window, 0, currentIndex, "SAME")

                true_prediction = unnormalize_np(correct_output[0], ndviMean, ndviStdDev)
                actual_prediction = unnormalize_np(roundArr, ndviMean, ndviStdDev)

                np.save("actualNDVI" + str(i), true_prediction)
                np.save("predictedNDVI" + str(i), actual_prediction)

                f2 = open("testResults.txt", "a")
                f2.write(str(rootmeansquare(true_prediction, actual_prediction)))
                f2.write("\n")

                sumSquareError = sumSquareError + (actual_prediction - true_prediction)**2

    sumSquareError = np.sqrt(sumSquareError/len(testList))
    finalValue = np.sum(sumSquareError)/10000
    print(str(finalValue))
    print(str(np.min(sumSquareError)))
    print(str(np.max(sumSquareError)))


def validate(validateList):
    global learning_rate
    global prev_validate

    global connected_weights
    global main_kernel
    global e_kernel
    global v_connected_weights
    global bias_h
    global bias_y
    global bias_e
    global bias_v

    e_kernel = cp.asarray(np.load('5e_kernelfinal3.npy'))
    v_connected_weights = cp.asarray(np.load('5v_connected_weightsfinal3.npy'))
    bias_e = cp.asarray(np.load('5bias_efinal3.npy'))
    connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
    main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
    bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
    bias_h = cp.asarray(np.load('5bias_hfinal3.npy'))
    bias_v = cp.asarray(np.load('5bias_vfinal3.npy'))

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
                #prev_output = satellite_images[folder][currentIndex + learning_window - 1]

                print(str(np.max(correct_output[0])) + " max NDVI")
                print(str(np.min(correct_output[1])) + " max rain")

                roundArr, p = return_forecast(input, learning_window, 0, currentIndex, "SAME")

                true_prediction = unnormalize_np(correct_output[0], ndviMean, ndviStdDev)
                actual_prediction = unnormalize_np(roundArr, ndviMean, ndviStdDev)
                #p2 = unnormalize_np(p[learning_window-1][0], ndviMean, ndviStdDev)

                f2 = open("validate1.txt", "a")
                f2.write(str(rootmeansquare(true_prediction, actual_prediction)))
                f2.write("\n")

                average += rootmeansquare(true_prediction, actual_prediction)
                sumSquareError = sumSquareError + (actual_prediction - true_prediction[0])**2


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
