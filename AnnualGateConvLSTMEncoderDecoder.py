# This model was used in the Google Science Fair "Annual Gate ConvLSTM"
# model. It utilizes a modification known as an "Annual Gate" that allows
# the LSTM to more strongly learn the annual repititions in the crop data.
# Furthermore, the encoder-decoder model means it can predict the next 32
# days of NDVI vegetation images. 

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
learning_rate = 0.001

# number of sets of images
S = 1

# number of images per sequence/set
T = 410

# dimensions of the image
M = 100
N = 100

# how far to look back at x tau
distance = 39
distance_forward = 14
learning_window = 15
prev_validate = 100
tau_len = distance_forward

channels_img_Decode = 1 
channels_img = 2 # antecedent NDVI and rain
channels_hidden = 24
channels_hidden_initial = channels_hidden
kernel_dimension = 5
kernel_dimension_g = 5
kernel_dimension_p = 5
channels_p = 1
pad_constant = 2
channels_img_output = 1

steps_ahead = 4

#this values are saved for producing the final output that must be displayed to the user
stdev = 0
mean = 0

satellite_images = np.empty([S, 711, channels_img, M, N])

#---------------------------------------He Normal Initialization-------------------------------------
#---------------------------------------------DECODER------------------------------------------------
r_v_connected_weights = 2*math.sqrt(6/(channels_hidden_initial*M*N + 1))
r_e_kernel = 2*math.sqrt(6/(channels_hidden_initial + (channels_img + channels_hidden_initial)*(kernel_dimension)*(kernel_dimension)))

v_connected_weights = cp.random.uniform(-r_v_connected_weights, r_v_connected_weights,(channels_hidden*M*N))
e_kernel = cp.random.uniform(-r_e_kernel, r_e_kernel, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))


r_kernel_tanh = math.sqrt(6/((channels_hidden_initial+channels_img_Decode)*(kernel_dimension)*(kernel_dimension) + channels_hidden_initial))
r_kernel_sigmoid = math.sqrt(6/((channels_hidden_initial+channels_img_Decode)*(kernel_dimension)*(kernel_dimension) + channels_hidden_initial))
r_kernel_g = math.sqrt(6/((channels_hidden_initial + channels_p)*(kernel_dimension_g)*(kernel_dimension_g) + channels_hidden_initial))
r_connected_weights =  math.sqrt(6/(channels_hidden_initial + 1))
r_connected_weights_tau = math.sqrt(1/(tau_len + M*N*channels_hidden_initial))

increaseRate = 1.8
i_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img_Decode + channels_hidden, kernel_dimension, kernel_dimension))
f_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img_Decode + channels_hidden, kernel_dimension, kernel_dimension))
a_kernel = increaseRate*cp.random.uniform(-r_kernel_tanh, r_kernel_tanh, (channels_hidden, channels_img_Decode + channels_hidden, kernel_dimension, kernel_dimension))
o_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img_Decode + channels_hidden, kernel_dimension, kernel_dimension))
n_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img_Decode + channels_hidden, kernel_dimension, kernel_dimension))
main_kernel = cp.concatenate((i_kernel, f_kernel, a_kernel, o_kernel, n_kernel))
connected_weights = 0.8*cp.random.normal(-r_connected_weights, r_connected_weights, (1, channels_hidden))
g_kernel = increaseRate*cp.random.uniform(-r_kernel_g, r_kernel_g, (channels_hidden, channels_p + channels_hidden, kernel_dimension_g, kernel_dimension_g))

#--------------------------------------ENCODER--------------------------------------
r_kernel_tanh = math.sqrt(6/((channels_hidden_initial+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden_initial))
r_kernel_sigmoid = math.sqrt(6/((channels_hidden_initial+channels_img)*(kernel_dimension)*(kernel_dimension) + channels_hidden_initial))
r_connected_weights = math.sqrt(6/(channels_hidden + 1))

a_kernel = cp.random.uniform(-r_kernel_tanh, r_kernel_tanh, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
i_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
f_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
o_kernel = cp.random.uniform(-r_kernel_sigmoid, r_kernel_sigmoid, (channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension))
connected_weightsEncode = cp.random.normal(-r_connected_weights, r_connected_weights, (1, channels_hidden))
main_kernelEncode = cp.concatenate((i_kernel, f_kernel, a_kernel, o_kernel))


#--------------------------------------BIAS INITIALIZATION-----------------------------
bias_cEncode = cp.zeros([channels_hidden, M, N])
bias_iEncode = cp.zeros([channels_hidden, M, N])
bias_fEncode = cp.ones([channels_hidden, M, N])
bias_oEncode = cp.zeros([channels_hidden, M, N])
bias_yEncode = cp.zeros([channels_img, M, N])

ndviMean = 0.66673688145266
ndviStdDev = 0.16560766237944935
rainMean = 0.19636724781555773
rainStdDev = 0.16560766237944935

del i_kernel
del f_kernel
del a_kernel
del o_kernel
del n_kernel

learning_rate = 0.0005
clip_threshold = 5

bias_c = cp.zeros([channels_hidden, M, N])
bias_i = cp.full([channels_hidden, M, N], 0)
bias_f = cp.full([channels_hidden, M, N], 1)
bias_o = cp.zeros([channels_hidden, M, N])
bias_y = cp.full([channels_img, M, N], 0)
bias_n = cp.zeros([channels_hidden, M, N])
bias_g = cp.zeros([channels_hidden, M, N])
bias_e =  cp.zeros([channels_hidden, M, N])
bias_v = cp.zeros([distance_forward])

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = cp.exp(x - cp.max(x))
    return e_x / cp.sum(e_x)

def softmax_derivative(x):
    return softmax(x)*(1 - softmax(x))

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

def rect_linear(arr):
    newArr = copy.deepcopy(arr)
    newArr[arr<0] = 0
    return newArr

def rect_linear_derivative(arr):
    newArr = cp.zeros(arr.shape)
    newArr[arr>0] = 1
    return newArr

# x[t] is the input
def encode(x,local_time2):

    # Input Gate
    i = cp.empty([local_time2, channels_hidden, M, N])

    # Forget Gate
    f = cp.empty([local_time2, channels_hidden, M, N])

    # Memory
    a = cp.empty([local_time2, channels_hidden, M, N])

    # Cell Gate
    c = cp.empty([local_time2 + 1, channels_hidden, M, N])
    c[-1] = cp.zeros([channels_hidden, M, N])
    # Output Gate
    o = cp.empty([local_time2, channels_hidden, M, N])

    # Hidden Unit
    h = cp.empty([local_time2 + 1, channels_hidden, M, N])
    h[-1] = cp.zeros([channels_hidden, M, N])
    # LSTM FORWARD PROPAGATION
    for t in range(local_time2):

        temporary = cp.concatenate((x[t], h[t - 1]), axis=0)
        temporary = temporary.reshape(1, channels_img + channels_hidden, M, N)

        i[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernelEncode[0:channels_hidden], b=None, pad=pad_constant)[0].data) + bias_iEncode)
        f[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernelEncode[channels_hidden:2*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_fEncode)
        a[t] = tanh(cp.asarray(F.convolution_2d(temporary, main_kernelEncode[2*channels_hidden:3*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_cEncode)
        c[t] = cp.multiply(f[t], c[t - 1]) + cp.multiply(i[t], a[t])
        o[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernelEncode[3*channels_hidden:4*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_oEncode)
        h[t] = cp.multiply(o[t], tanh(c[t]))

    # 1 x 1 convolution
    #output = cp.matmul(connected_weights, h[local_time-1].reshape(channels_hidden, M * N)).reshape(M, N) + bias_y[0]
    print("CONNECTED_WEIGHTS NORM: " + str(cp.linalg.norm(connected_weights)))
    print("HIDDEN_PREDICTION NORM: " + str(cp.linalg.norm(h[local_time2-1])))
    return cp.reshape(h[local_time2-1], (channels_hidden, M*N)), i, f, a, c, o, h

# x[t] is the input
def decode(local_time2, sequence, isFirst, timestamp, satellite_name, cEncode, hEncode, initial_input_Image):
    global bias_tau

    # Input Gate
    i = cp.empty([local_time2, channels_hidden, M, N])

    # Forget Gate
    f = cp.empty([local_time2, channels_hidden, M, N])

    # Memory
    a = cp.empty([local_time2, channels_hidden, M, N])

    # Output Gate
    o = cp.empty([local_time2, channels_hidden, M, N])

    n = cp.empty([local_time2, channels_hidden, M, N])
    p = cp.empty([local_time2, channels_p, M, N])
    g = cp.empty([local_time2, channels_hidden, M, N])
    s = cp.empty([local_time2, distance_forward, channels_hidden, M, N])
    e = cp.empty([local_time2, distance_forward])

    alpha = cp.empty([local_time2, distance_forward])
    initial_Image = initial_input_Image
    xDecode = cp.zeros([local_time2, M, N])
    
    # Cell Gate
    c = cp.empty([local_time2 + 1, channels_hidden, M, N])
    c[-1] = cEncode

    # Hidden Unit
    h = cp.empty([local_time2 + 1, channels_hidden, M, N])
    h[-1] = hEncode

    hidden_prediction = cp.empty([local_time2, channels_hidden, M, N])

    actual_output = cp.zeros([local_time2, M, N])
    actual_output_pre_sigmoid = cp.zeros([local_time2, M, N]) 
    # LSTM FORWARD PROPAGATION
    for t in range(local_time2):

        temporary = cp.concatenate((initial_Image.reshape(1, M, N), h[t - 1]), axis=0)
        temporary = temporary.reshape(1, channels_img_output + channels_hidden, M, N)
        xDecode[t] = initial_Image[0]
        print("HIDDEN PREDICTION INPUT: " + str(cp.mean(h[t - 1])))
        print("IMAGE INPUT: " + str(cp.mean(initial_Image)))
        i[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernel[0:channels_hidden], b=None, pad=pad_constant)[0].data) + bias_i)

        f[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernel[channels_hidden:2*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_f)

        a[t] = tanh(cp.asarray(F.convolution_2d(temporary, main_kernel[2*channels_hidden:3*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_c)

        n[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernel[4*channels_hidden:5*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_n)

        # Attention Network
        for z in range(timestamp + t - (distance + learning_window), timestamp + distance_forward + t - (distance + learning_window)):
            temp = cp.concatenate((cp.asarray(satellite_images[sequence][z]), h[t - 1]), axis = 0)
            s[t][z - (timestamp + t - (distance + learning_window))] = tanh(cp.asarray(F.convolution_2d(temp.reshape(1, channels_img + channels_hidden, M, N), e_kernel, b=None, pad=pad_constant)[0].data) + bias_e)
            s_temp = s[t][z - (timestamp + t - (distance + learning_window))].reshape(M*N*channels_hidden)
            e[t][z - (timestamp + t - (distance + learning_window))] = cp.dot(v_connected_weights, s_temp) + bias_v[z - (timestamp + t - (distance + learning_window))]

        xtemp = satellite_images[sequence][timestamp - distance:timestamp-distance+distance_forward, 0]

        alpha[t] = softmax(e[t])
        p[t] = cp.tensordot(alpha[t], cp.asarray(xtemp), axes = 1).reshape(1, M, N) # Sum all x arrays up, weighted array

        temporary2 = cp.concatenate((p[t], h[t-1]), axis = 0)
        temporary2 = temporary2.reshape(1, channels_p + channels_hidden, M, N)

        g[t] = tanh(cp.asarray(F.convolution_2d(temporary2, g_kernel, b=None, pad=pad_constant)[0].data) + bias_g)

        c[t] = cp.multiply(f[t], c[t - 1]) + cp.multiply(i[t], a[t]) + cp.multiply(n[t], g[t])

        o[t] = sigmoid(cp.asarray(F.convolution_2d(temporary, main_kernel[3*channels_hidden:4*channels_hidden], b=None, pad=pad_constant)[0].data) + bias_o)

        h[t] = cp.multiply(o[t], tanh(c[t]))

        output = cp.matmul(connected_weights, h[t].reshape(channels_hidden, M * N)).reshape(M, N) + bias_y[0]
        true_output = sigmoid(output)
        actual_output[t] = true_output
        actual_output_pre_sigmoid[t] = output  
        initial_Image = true_output.reshape(1, M, N)

      
    return actual_output, actual_output_pre_sigmoid, cp.reshape(h[local_time2-1], (channels_hidden, M*N)), i, f, a, c, o, h, n, g, p, s, e, alpha, xtemp, xDecode

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

def return_forecast(x, local_time, timestamp):
    hidden_predictionEncode, iEncode, fEncode, aEncode, cEncode, oEncode, hEncode = encode(x, local_time-1)
    prediction, pre_sigmoid_prediction, hidden_prediction, i, f, a, c, o, h, n, g, p, s, e, alpha, xtemp, xDecode = decode(steps_ahead, 0, False, timestamp + local_time - 1, "SAME", cEncode[-2], hEncode[-2],  x[len(x)-1][0])

    return cp.asnumpy(prediction)

def loss_derivative(x, y):
    return (x-y)
    
def bptt(x2, y2, iteration, local_time, region, isFirst, timestamp, satellite_name):
    #--------------------------------------------DECODER BACKPROP CODE---------------------------------------------
    x = cp.asarray(x2)
    y = cp.asarray(y2)

    global connected_weights
    global main_kernel
    global bias_i
    global bias_f
    global bias_c
    global bias_o
    global bias_y
    global bias_n
    global bias_g
    global e_kernel
    global bias_e
    global learning_rate
    global g_kernel
    global v_connected_weights
    global bias_v


    # Perform forward prop
    hidden_predictionEncode, iEncode, fEncode, aEncode, cEncode, oEncode, hEncode = encode(x, local_time-1)
    prediction, pre_sigmoid_prediction, hidden_prediction, i, f, a, c, o, h, n, g, p, s, e, alpha, xtemp, xDecode = decode(steps_ahead, region, isFirst, timestamp + local_time - 1, satellite_name, cEncode[-2], hEncode[-2],  x[len(x)-1][0])

    sumLossInitial = 0
    for counter in range(0, steps_ahead):
        loss = calculate_loss(prediction[counter], y[counter:counter+1, 0][0])
        sumLossInitial += loss
        print("LOSS " + str(counter+1) + ": " + str(loss))
        if loss > 100:
            f2 = open("wrong.txt", "a")
            f2.write(str(timestamp) + "\n")

    #---------------------------------------------DECODER DERIVATIVE COMPUTATION--------------------------------------------
    
    dLdW = cp.zeros([5*channels_hidden, channels_img_Decode + channels_hidden, kernel_dimension, kernel_dimension])

    # Initialize other weight matrices
    dLdW_g = cp.zeros([channels_hidden, channels_p + channels_hidden, kernel_dimension_g, kernel_dimension_g])
    dLdW_v = cp.zeros([channels_hidden*M*N])
    dLdW_e = cp.zeros([channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension])
    dLdw_0 = cp.zeros(connected_weights.shape)

    # initialize biases
    dLdb_c = cp.zeros([channels_hidden, M, N])
    dLdb_i = cp.zeros([channels_hidden, M, N])
    dLdb_f = cp.zeros([channels_hidden, M, N])
    dLdb_o = cp.zeros([channels_hidden, M, N])
    dLdb_n = cp.zeros([channels_hidden, M, N])
    dLdb_g = cp.zeros([channels_hidden, M, N])
    dLdb_e = cp.zeros([channels_hidden, M, N])
    dLdb_v = cp.zeros([distance_forward])
    dLdb_y = cp.zeros(bias_y.shape)

    # Initialize cell matrix
    dLdc_current = cp.zeros([channels_hidden, M, N])
    dLdx = cp.zeros([M, N])
    dLdh = cp.zeros([channels_hidden, M, N])
    for t in range(steps_ahead - 1, -1, -1):
        dLdy_2 = dLdx + loss_derivative(prediction[t], y[t:t+1, 0])
        # Might NEED TO CHANGE pre_sigmoid_prediction, note that there is 
        # bracket [t] to indicate timestep
        dLdy_1 = cp.multiply(sigmoid_derivative(pre_sigmoid_prediction[t]), dLdy_2)
        testArr = cp.reshape(cp.matmul(cp.transpose(connected_weights), dLdy_1.reshape(1, M * N)), (channels_hidden, M, N))

        dLdh = testArr + dLdh
        # bracket [t] to indicate timestep
        dLdw_0 += cp.matmul(dLdy_1.reshape(1, M*N), h[t].reshape(channels_hidden, M*N).transpose(1,0))
        dLdb_y += dLdy_1

        dLdo = cp.multiply(dLdh, tanh(c[t]))
        dLdc_current += cp.multiply(cp.multiply(dLdh, o[t]), (cp.ones((channels_hidden, M, N)) - cp.multiply(tanh(c[t]), tanh(c[t]))))
        dLdi = cp.multiply(dLdc_current, a[t])
        dLda = cp.multiply(dLdc_current, i[t])
        dLdf = cp.multiply(dLdc_current, c[t - 1])

        dLdg = cp.multiply(dLdc_current, n[t])
        dLdn = cp.multiply(dLdc_current, g[t])

        dLdc_previous = cp.multiply(dLdc_current, f[t])
        dLda = cp.multiply(dLda, (cp.ones((channels_hidden, M, N)) - cp.multiply(a[t], a[t]))) #dLda_hat
        dLdi = cp.multiply(cp.multiply(dLdi, i[t]), cp.ones((channels_hidden, M, N)) - i[t]) #dLdi_hat
        dLdf = cp.multiply(cp.multiply(dLdf, f[t]), cp.ones((channels_hidden, M, N)) - f[t]) #dLdf_hat
        dLdo = cp.multiply(cp.multiply(dLdo, o[t]), cp.ones((channels_hidden, M, N)) - o[t]) #dLdo_hat
        dLdg = cp.multiply(dLdg, (cp.ones((channels_hidden, M, N)) - cp.multiply(g[t], g[t]))) #dLdg_hat
        dLdn = cp.multiply(cp.multiply(dLdn, n[t]), cp.ones((channels_hidden, M, N)) - n[t]) #dLdn_hat

        temporary_p = cp.concatenate((p[t], h[t - 1]), axis=0).reshape(channels_hidden + channels_p, 1, M, N)

        #Convolve to continue backpropagation along the annual gate path
        dLdI_g = cp.asarray(F.convolution_2d(dLdg.reshape(1, channels_hidden, M, N), g_kernel.transpose(1, 0, 2, 3), b=None, pad=pad_constant)[0].data)
        dLdW_g_temp = cp.asarray((F.convolution_2d(temporary_p, dLdg.reshape(channels_hidden, 1, M, N), b=None, pad=pad_constant).data).transpose(1,0,2,3))
        #create dLdp, which is the derivative of loss with respect to p
        dLdp = dLdI_g[0: channels_p]
        dLdAlpha = cp.zeros(distance_forward)
        for k in range(0, distance_forward):
            dLdAlpha[k] = cp.sum(dLdp*cp.asarray(xtemp[k]))

        dLde = cp.zeros(distance_forward)
        for z in range(distance_forward):
            for j in range(distance_forward):
                if z == j:
                    dLde[z] += dLdAlpha[j] * alpha[t][j] * (1 - alpha[t][z])
                else:
                    dLde[z] += -dLdAlpha[j] * alpha[t][j] * alpha[t][z]

        dLdW_v_temp = cp.zeros([channels_hidden*M*N])
        dLdW_e_temp = cp.zeros([channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension])
        dLdh_temp = cp.zeros([channels_hidden, M, N])

        for k in range(0, distance_forward):
            dLdW_v_temp += dLde[k]*s[t][k].reshape((M*N*channels_hidden))
            dLds = dLde[k]*v_connected_weights # Changes each iteration of nested loop
            dLds = dLds.reshape((channels_hidden, M, N))
            dLds = cp.multiply(dLds, (cp.ones((channels_hidden, M, N)) - cp.multiply(s[t][k], s[t][k])))
            temp3 = cp.concatenate((cp.asarray(satellite_images[region][int(k - (timestamp + t - (distance + learning_window)))]), h[t - 1]), axis = 0)


            dLdI_e = cp.asarray(F.convolution_2d(dLds.reshape(1, channels_hidden, M, N), e_kernel.transpose(1, 0, 2, 3), b=None, pad=pad_constant)[0].data) # reshape into flipped kernel dimensions
            dLdW_e_temp += cp.asarray((F.convolution_2d(temp3.reshape(channels_hidden+channels_img, 1, M, N), dLds.reshape(channels_hidden, 1, M, N), b=None, pad=pad_constant).data).transpose(1,0,2,3)) #reshape into kernel dimensions
            dLdh_temp += dLdI_e[channels_img: channels_img + channels_hidden]
            # if cp.amax(dLds) > 1 or cp.amin(dLds) < -1:
            #     dLds = dLds/cp.linalg.norm(dLds)
            dLdb_e += dLds

        dLdb_c += dLda
        dLdb_i += dLdi
        dLdb_f += dLdf
        dLdb_o += dLdo
        dLdb_n += dLdn
        dLdb_g += dLdg
        

        # CONCATENATE Z IN THE RIGHT ORDER SAME ORDER AS THE WEIGHTS
        dLdz_hat = cp.concatenate((dLdi, dLdf, dLda, dLdo, dLdn), axis = 0) 
        del dLdi
        del dLdf
        del dLda 
        del dLdo
        del dLdn
        del dLds
        del dLdg

        #determine convolution derivatives (main convolution)
        #here we will use the fact that in z = w * I, dLdW = dLdz * I
        temporary = cp.concatenate((xDecode[t].reshape(1, M, N), h[t - 1]), axis=0).reshape(channels_hidden + channels_img_Decode, 1, M, N)
        dLdW_temp = cp.asarray((F.convolution_2d(temporary, dLdz_hat.reshape(5*channels_hidden, 1, M, N), b=None, pad=pad_constant).data).transpose(1,0,2,3)) #reshape into kernel dimensions

        
        # accumulate derivatives of weights and biases
        dLdW += dLdW_temp 
        dLdW_g += dLdW_g_temp
        dLdW_e += dLdW_e_temp
        dLdW_v += dLdW_v_temp
        dLdb_v += dLde.reshape([distance_forward])

        del dLdW_temp
        del dLdW_g_temp
        del dLdW_v_temp
        del dLdW_e_temp
        del dLde

        dLdI = (F.convolution_2d(cp.asnumpy(dLdz_hat).reshape(1, 5*channels_hidden, M, N), cp.asnumpy(main_kernel).transpose(1, 0, 2, 3), b=None, pad=pad_constant)[0].data) # reshape into flipped kernel dimensions
        dLdx = cp.asarray(dLdI[0: channels_img_Decode])
        # reinitialize what you're passing back
        dLdh = cp.asarray(dLdI[channels_img_Decode: channels_img_Decode+channels_hidden]) + dLdI_g[channels_p: channels_p+channels_hidden] + dLdh_temp
        dLdc_current = dLdc_previous

    #delete variables from RAM 
    del prediction 
    del pre_sigmoid_prediction 
    del hidden_prediction
    del i
    del f
    del a
    del c
    del o
    del h
    del n
    del g
    del p
    del s
    del e
    del alpha
    del xtemp

    #Gradient clipping
    if cp.linalg.norm(dLdW) > clip_threshold:
        dLdW = dLdW*clip_threshold/cp.linalg.norm(dLdW)
    if cp.linalg.norm(dLdW_g) > clip_threshold:
        dLdW_g = dLdW_g*clip_threshold/cp.linalg.norm(dLdW_g)
    if cp.linalg.norm(dLdW_e) > clip_threshold:
        dLdW_e = dLdW_e*clip_threshold/cp.linalg.norm(dLdW_e)
    if cp.linalg.norm(dLdW_v) > clip_threshold:
        dLdW_v = dLdW_v*clip_threshold/cp.linalg.norm(dLdW_v)
    if cp.linalg.norm(dLdb_c) > clip_threshold:
        dLdb_c = dLdb_c*clip_threshold/cp.linalg.norm(dLdb_c)
    if cp.linalg.norm(dLdb_i) > clip_threshold:
        dLdb_i = dLdb_i*clip_threshold/cp.linalg.norm(dLdb_i)
    if cp.linalg.norm(dLdb_f) > clip_threshold:
        dLdb_f = dLdb_f*clip_threshold/cp.linalg.norm(dLdb_f)
    if cp.linalg.norm(dLdb_o) > clip_threshold:
        dLdb_o = dLdb_o*clip_threshold/cp.linalg.norm(dLdb_o)
    if cp.linalg.norm(dLdb_n) > clip_threshold:
        dLdb_n = dLdb_n*clip_threshold/cp.linalg.norm(dLdb_n)
    if cp.linalg.norm(dLdb_g) > clip_threshold:
        dLdb_g = dLdb_g*clip_threshold/cp.linalg.norm(dLdb_g)
    if cp.linalg.norm(dLdb_e) > clip_threshold:
        dLdb_e = dLdb_e*clip_threshold/cp.linalg.norm(dLdb_e)
    if cp.linalg.norm(dLdb_v) > clip_threshold:
        dLdb_v = dLdb_v*clip_threshold/cp.linalg.norm(dLdb_v)
    
    #---------------------update main kernel---------
    main_kernel = main_kernel - learning_rate*dLdW
    #---------------------update g kernel---------
    g_kernel = g_kernel - learning_rate*dLdW_g
    #---------------------update e kernel---------
    e_kernel = e_kernel - learning_rate*dLdW_e
    #---------------------update v_connected_weights---------
    v_connected_weights = v_connected_weights - learning_rate*dLdW_v
    #--------------------update bias c-----------------------
    bias_c = bias_c - learning_rate*dLdb_c
    #--------------------update bias i-----------------------
    bias_i = bias_i - learning_rate*dLdb_i
    #--------------------update bias f-----------------------
    bias_f = bias_f - learning_rate*dLdb_f
    #--------------------update bias o-----------------------
    bias_o = bias_o - learning_rate*dLdb_o
    #--------------------update bias n-----------------------
    bias_n = bias_n - learning_rate*dLdb_n
    #--------------------update bias g-----------------------
    bias_g = bias_g - learning_rate*dLdb_g
    #--------------------update bias e-----------------------
    bias_e = bias_e - learning_rate*bias_e
    #--------------------update bias v-----------------------
    bias_v = bias_v - learning_rate*dLdb_v
    #--------------------update connected_weights------------
    connected_weights = connected_weights - learning_rate*dLdw_0
    #--------------------update bias y-----------------------
    bias_y = bias_y - learning_rate*dLdb_y

    #------------------------------------------LSTM ENCODER BACKPROPAGATION CODE-----------------------------------------------
    global connected_weightsEncode
    global main_kernelEncode
    global bias_iEncode
    global bias_fEncode
    global bias_cEncode
    global bias_oEncode
    global bias_yEncode

    # Initialize weight matrix
    dLdW = cp.zeros([4*channels_hidden, channels_img + channels_hidden, kernel_dimension, kernel_dimension])

    # initialize biases
    dLdb_c = cp.zeros([channels_hidden, M, N])
    dLdb_i = cp.zeros([channels_hidden, M, N])
    dLdb_f = cp.zeros([channels_hidden, M, N])
    dLdb_o = cp.zeros([channels_hidden, M, N])

    # Initialize cell matrix
    #dLdc_current = cp.zeros([channels_hidden, M, N])

    for t in range(local_time - 2, -1, -1):
        dLdo = cp.multiply(dLdh, tanh(cEncode[t]))
        dLdc_current += cp.multiply(cp.multiply(dLdh, oEncode[t]), (cp.ones((channels_hidden, M, N)) - cp.multiply(tanh(cEncode[t]), tanh(cEncode[t]))))
        dLdi = cp.multiply(dLdc_current, aEncode[t])
        dLda = cp.multiply(dLdc_current, iEncode[t])
        dLdf = cp.multiply(dLdc_current, cEncode[t - 1])

        dLdc_previous = cp.multiply(dLdc_current, fEncode[t])

        dLda = cp.multiply(dLda, (cp.ones((channels_hidden, M, N)) - cp.multiply(aEncode[t], aEncode[t]))) #dLda_hat

        dLdi = cp.multiply(cp.multiply(dLdi, iEncode[t]), cp.ones((channels_hidden, M, N)) - iEncode[t]) #dLdi_hat

        dLdf = cp.multiply(cp.multiply(dLdf, fEncode[t]), cp.ones((channels_hidden, M, N)) - fEncode[t]) #dLdf_hat

        dLdo = cp.multiply(cp.multiply(dLdo, oEncode[t]), cp.ones((channels_hidden, M, N)) - oEncode[t]) #dLdo_hat


        # CONCATENATE Z IN THE RIGHT ORDER SAME ORDER AS THE WEIGHTS
        dLdz_hat = cp.concatenate((dLdi, dLdf, dLda, dLdo), axis = 0) 

        #determine convolution derivatives
        #here we will use the fact that in z = w * I, dLdW = dLdz * I
        temporary = cp.concatenate((x[t], hEncode[t - 1]), axis=0).reshape(channels_hidden + channels_img, 1, M, N)
        dLdI = cp.asarray(F.convolution_2d(dLdz_hat.reshape(1, 4*channels_hidden, M, N), main_kernelEncode.transpose(1, 0, 2, 3), b=None, pad=pad_constant)[0].data) # reshape into flipped kernel dimensions
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

    #Clip all gradients again
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
    main_kernelEncode = main_kernelEncode - learning_rate*dLdW
    #--------------------update bias c-----------------------
    bias_cEncode = bias_cEncode - learning_rate*dLdb_c
    #--------------------update bias i-----------------------
    bias_iEncode = bias_iEncode - learning_rate*dLdb_i
    #--------------------update bias f-----------------------
    bias_fEncode = bias_fEncode - learning_rate*dLdb_f
    #--------------------update bias c-----------------------
    bias_oEncode = bias_oEncode - learning_rate*dLdb_o

    # Perform forward prop
    hidden_predictionEncode, iEncode, fEncode, aEncode, cEncode, oEncode, hEncode = encode(x, local_time-1)
    prediction, pre_sigmoid_prediction, hidden_prediction, i, f, a, c, o, h, n, g, p, s, e, alpha, xtemp, xDecode = decode(steps_ahead, region, isFirst, timestamp + local_time - 1, satellite_name, cEncode[-2], hEncode[-2],  x[len(x)-1][0])

    sumLossFinal = 0
    for counter in range(0, steps_ahead):
        loss = calculate_loss(prediction[counter], y[counter:counter+1, 0][0])
        sumLossFinal += loss
        print("LOSS AFTER" + str(counter+1) + ": " + str(loss))
        #----------------PRINT RMSE-----------------
        rms = rootmeansquare(unnormalize_cp(prediction[counter], ndviMean, ndviStdDev), unnormalize_cp(y[counter:counter+1, 0], ndviMean, ndviStdDev))
        f2 = open("loss" + str(counter+1) + ".txt", "a")
        f2.write(str(rms) + "\n")
    
    if sumLossFinal > sumLossInitial:
        print("-----------------------------------Close----------------------------------")
        #learning_rate = learning_rate*0.8

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

    list1 = produceRandomImageArray()

    main(list1)


def schedule(i):
    return 0.0002

def MAPE(correct, prediction):
    return np.sum(np.absolute(correct-prediction)/correct)/100
    correct[correct == 0] = 0.000001
    prediction[prediction == 0] = 0.000001

def main(indexGeneralList):
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

    indexList = indexGeneralList[0:439] 
    validateList = indexGeneralList[439:564]
    testList = indexGeneralList[564:627]
    

    for e in range(0, 10):
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

                    correct_output = satellite_images[folder][currentIndex+learning_window: currentIndex + learning_window + steps_ahead]
                    print(correct_output.shape)

                    first = False
                    if i == 0:
                        first = True

                    bptt(input, correct_output, i, learning_window, folder, first, currentIndex, "TERRA")

            if i%50 == 0:
                print("-------------------Weight Matrix----------------")
                np.save('5main_kernelfinal3', cp.asnumpy(main_kernel))
                print("------------------connected_weights---------------------")
                np.save('5connected_weightsfinal3', cp.asnumpy(connected_weights))
                print("-------------------g_kernel-------------------------")
                np.save('5g_kernelfinal3', cp.asnumpy(g_kernel))
                print("-------------------------bias_g--------------------")
                np.save('5bias_gfinal3', cp.asnumpy(bias_g))
                print("-------------------------bias_n--------------------")
                np.save('5bias_nfinal3', cp.asnumpy(bias_n))
                print("-------------------bias_y-------------------------")
                np.save('5bias_yfinal3', cp.asnumpy(bias_y))
                print("----------------------bias_o-----------------------")
                np.save('5bias_ofinal3', cp.asnumpy(bias_o))
                print("-------------------bias_c-------------------------")
                np.save('5bias_cfinal3', cp.asnumpy(bias_c))
                print("----------------------bias_f------------------")
                np.save('5bias_ffinal3', cp.asnumpy(bias_f))
                print("-----------------------bias_i-------------------")
                np.save('5bias_ifinal3', cp.asnumpy(bias_i))
                print("-----------------------v_connected_weights-------------------")
                np.save('5v_connected_weightsfinal3', cp.asnumpy(v_connected_weights))
                print("----------------------bias_o-----------------------")
                np.save('5bias_oEncodefinal3', cp.asnumpy(bias_oEncode))
                print("-------------------bias_c-------------------------")
                np.save('5bias_cEncodefinal3', cp.asnumpy(bias_cEncode))
                print("----------------------bias_f------------------")
                np.save('5bias_fEncodefinal3', cp.asnumpy(bias_fEncode))
                print("-----------------------bias_i-------------------")
                np.save('5bias_iEncodefinal3', cp.asnumpy(bias_iEncode))
                print("-------------------Weight Matrix----------------")
                np.save('5main_kernelEncodefinal3', cp.asnumpy(main_kernelEncode))
                        
        validate(validateList)
    test(testList)

def produceRandomImageArray():
    list = []
    for i in range(55, 682):
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

    global bias_n
    global bias_g
    global g_kernel
    global v_connected_weights

    global bias_oEncode
    global bias_cEncode
    global bias_fEncode
    global bias_iEncode
    global main_kernelEncode


    connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
    main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
    bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
    bias_o = cp.asarray(np.load('5bias_ofinal3.npy'))
    bias_c = cp.asarray(np.load('5bias_cfinal3.npy'))
    bias_f = cp.asarray(np.load('5bias_ffinal3.npy'))
    bias_i = cp.asarray(np.load('5bias_ifinal3.npy'))

    bias_n = cp.asarray(np.load('5bias_nfinal3.npy'))
    bias_g = cp.asarray(np.load('5bias_gfinal3.npy'))
    g_kernel = cp.asarray(np.load('5g_kernelfinal3.npy'))
    v_connected_weights = cp.asarray(np.load('5v_connected_weightsfinal3.npy'))

    bias_oEncode = cp.asarray(np.load('5bias_oEncodefinal3.npy'))
    bias_cEncode = cp.asarray(np.load('5bias_cEncodefinal3.npy'))
    bias_fEncode = cp.asarray(np.load('5bias_fEncodefinal3.npy'))
    bias_iEncode = cp.asarray(np.load('5bias_iEncodefinal3.npy'))
    main_kernelEncode = cp.asarray(np.load('5main_kernelEncodefinal3.npy'))

    global learning_rate
    global prev_validate

    average = 0

    sumSquareError1 = np.zeros([M,N])
    sumSquareError2 = np.zeros([M,N])
    sumSquareError3 = np.zeros([M,N])
    sumSquareError4 = np.zeros([M,N])
    
    for i in range (0, len(testList)):
        #folder = random.randint(0, 8)
        imageSatCurrent = testList[i]
        folder = 0
        currentIndex = imageSatCurrent.index
        if imageSatCurrent.satellite == "SAME":
            if currentIndex + learning_window + steps_ahead< len(satellite_images[folder]):
                input = satellite_images[folder][currentIndex:(currentIndex+learning_window)]

                correct_output1 = satellite_images[folder][currentIndex+learning_window]
                correct_output2 = satellite_images[folder][currentIndex+learning_window + 1]
                correct_output3 = satellite_images[folder][currentIndex+learning_window + 2]
                correct_output4 = satellite_images[folder][currentIndex+learning_window + 3]

                print(str(np.max(correct_output1[0])) + " max NDVI")
                print(str(np.min(correct_output1[1])) + " max rain")

                roundArr = return_forecast(cp.asarray(input), learning_window, currentIndex)
            
                true_prediction1 = unnormalize_np(correct_output1[0], ndviMean, ndviStdDev)
                actual_prediction1 = unnormalize_np(roundArr[0], ndviMean, ndviStdDev)

                true_prediction2 = unnormalize_np(correct_output2[0], ndviMean, ndviStdDev)
                actual_prediction2 = unnormalize_np(roundArr[1], ndviMean, ndviStdDev)

                true_prediction3 = unnormalize_np(correct_output3[0], ndviMean, ndviStdDev)
                actual_prediction3 = unnormalize_np(roundArr[2], ndviMean, ndviStdDev)

                true_prediction4 = unnormalize_np(correct_output4[0], ndviMean, ndviStdDev)
                actual_prediction4 = unnormalize_np(roundArr[3], ndviMean, ndviStdDev)

                f2 = open("test1.txt", "a")
                f2.write(str(rootmeansquare(true_prediction1, actual_prediction1)))
                f2.write("\n")

                f2 = open("test2.txt", "a")
                f2.write(str(rootmeansquare(true_prediction2, actual_prediction2)))
                f2.write("\n")

                f2 = open("test3.txt", "a")
                f2.write(str(rootmeansquare(true_prediction3, actual_prediction3)))
                f2.write("\n")

                f2 = open("test4.txt", "a")
                f2.write(str(rootmeansquare(true_prediction4, actual_prediction4)))
                f2.write("\n")                

                average += rootmeansquare(true_prediction1, actual_prediction1) + rootmeansquare(true_prediction2, actual_prediction2) + rootmeansquare(true_prediction3, actual_prediction3) + rootmeansquare(true_prediction4, actual_prediction4)

                sumSquareError1 = sumSquareError1 + (true_prediction1 - actual_prediction1)**2
                sumSquareError2 = sumSquareError2 + (true_prediction2 - actual_prediction2)**2
                sumSquareError3 = sumSquareError3 + (true_prediction3 - actual_prediction3)**2
                sumSquareError4 = sumSquareError4 + (true_prediction4 - actual_prediction4)**2

    average = average/(137*4)
    if average>prev_validate:
        learning_rate = learning_rate * 0.8

    prev_validate = average

    sumSquareError1 = np.sqrt(sumSquareError1/len(validateList))
    finalValue1 = np.sum(sumSquareError1)/10000

    sumSquareError2 = np.sqrt(sumSquareError2/len(validateList))
    finalValue2 = np.sum(sumSquareError2)/10000

    sumSquareError3 = np.sqrt(sumSquareError3/len(validateList))
    finalValue3 = np.sum(sumSquareError3)/10000

    sumSquareError4 = np.sqrt(sumSquareError4/len(validateList))
    finalValue4 = np.sum(sumSquareError4)/10000

    f2 = open("test21.txt", "a")
    f2.write(str(finalValue1) + "\n")
    f2.write(str(np.min(sumSquareError1)) + "\n")
    f2.write(str(np.max(sumSquareError1)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("test22.txt", "a")
    f2.write(str(finalValue2) + "\n")
    f2.write(str(np.min(sumSquareError2)) + "\n")
    f2.write(str(np.max(sumSquareError2)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("test23.txt", "a")
    f2.write(str(finalValue3) + "\n")
    f2.write(str(np.min(sumSquareError3)) + "\n")
    f2.write(str(np.max(sumSquareError3)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("test24.txt", "a")
    f2.write(str(finalValue4) + "\n")
    f2.write(str(np.min(sumSquareError4)) + "\n")
    f2.write(str(np.max(sumSquareError4)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("test1.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("test2.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("test3.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("test4.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")


def validate(validateList):
    global connected_weights
    global main_kernel
    global bias_i
    global bias_f
    global bias_c
    global bias_o
    global bias_y

    global bias_n
    global bias_g
    global g_kernel
    global v_connected_weights

    global bias_oEncode
    global bias_cEncode
    global bias_fEncode
    global bias_iEncode
    global main_kernelEncode


    connected_weights = cp.asarray(np.load('5connected_weightsfinal3.npy'))
    main_kernel = cp.asarray(np.load('5main_kernelfinal3.npy'))
    bias_y = cp.asarray(np.load('5bias_yfinal3.npy'))
    bias_o = cp.asarray(np.load('5bias_ofinal3.npy'))
    bias_c = cp.asarray(np.load('5bias_cfinal3.npy'))
    bias_f = cp.asarray(np.load('5bias_ffinal3.npy'))
    bias_i = cp.asarray(np.load('5bias_ifinal3.npy'))

    bias_n = cp.asarray(np.load('5bias_nfinal3.npy'))
    bias_g = cp.asarray(np.load('5bias_gfinal3.npy'))
    g_kernel = cp.asarray(np.load('5g_kernelfinal3.npy'))
    v_connected_weights = cp.asarray(np.load('5v_connected_weightsfinal3.npy'))

    bias_oEncode = cp.asarray(np.load('5bias_oEncodefinal3.npy'))
    bias_cEncode = cp.asarray(np.load('5bias_cEncodefinal3.npy'))
    bias_fEncode = cp.asarray(np.load('5bias_fEncodefinal3.npy'))
    bias_iEncode = cp.asarray(np.load('5bias_iEncodefinal3.npy'))
    main_kernelEncode = cp.asarray(np.load('5main_kernelEncodefinal3.npy'))

    global learning_rate
    global prev_validate

    average = 0

    sumSquareError1 = np.zeros([M,N])
    sumSquareError2 = np.zeros([M,N])
    sumSquareError3 = np.zeros([M,N])
    sumSquareError4 = np.zeros([M,N])
    
    for i in range (0, len(validateList)):
        #folder = random.randint(0, 8)
        imageSatCurrent = validateList[i]
        folder = 0
        currentIndex = imageSatCurrent.index
        if imageSatCurrent.satellite == "SAME":
            if currentIndex + learning_window + 4 < len(satellite_images[folder]):
                input = satellite_images[folder][currentIndex:(currentIndex+learning_window)]

                correct_output1 = satellite_images[folder][currentIndex+learning_window]
                correct_output2 = satellite_images[folder][currentIndex+learning_window + 1]
                correct_output3 = satellite_images[folder][currentIndex+learning_window + 2]
                correct_output4 = satellite_images[folder][currentIndex+learning_window + 3]

                print(str(np.max(correct_output1[0])) + " max NDVI")
                print(str(np.min(correct_output1[1])) + " max rain")

                roundArr = return_forecast(cp.asarray(input), learning_window, currentIndex)
            
                true_prediction1 = unnormalize_np(correct_output1[0], ndviMean, ndviStdDev)
                actual_prediction1 = unnormalize_np(roundArr[0], ndviMean, ndviStdDev)

                true_prediction2 = unnormalize_np(correct_output2[0], ndviMean, ndviStdDev)
                actual_prediction2 = unnormalize_np(roundArr[1], ndviMean, ndviStdDev)

                true_prediction3 = unnormalize_np(correct_output3[0], ndviMean, ndviStdDev)
                actual_prediction3 = unnormalize_np(roundArr[2], ndviMean, ndviStdDev)

                true_prediction4 = unnormalize_np(correct_output4[0], ndviMean, ndviStdDev)
                actual_prediction4 = unnormalize_np(roundArr[3], ndviMean, ndviStdDev)

                f2 = open("validate1.txt", "a")
                f2.write(str(rootmeansquare(true_prediction1, actual_prediction1)))
                f2.write("\n")

                f2 = open("validate2.txt", "a")
                f2.write(str(rootmeansquare(true_prediction2, actual_prediction2)))
                f2.write("\n")

                f2 = open("validate3.txt", "a")
                f2.write(str(rootmeansquare(true_prediction3, actual_prediction3)))
                f2.write("\n")

                f2 = open("validate4.txt", "a")
                f2.write(str(rootmeansquare(true_prediction4, actual_prediction4)))
                f2.write("\n")                

                average += rootmeansquare(true_prediction1, actual_prediction1) + rootmeansquare(true_prediction2, actual_prediction2) + rootmeansquare(true_prediction3, actual_prediction3) + rootmeansquare(true_prediction4, actual_prediction4)

                sumSquareError1 = sumSquareError1 + (true_prediction1 - actual_prediction1)**2
                sumSquareError2 = sumSquareError2 + (true_prediction2 - actual_prediction2)**2
                sumSquareError3 = sumSquareError3 + (true_prediction3 - actual_prediction3)**2
                sumSquareError4 = sumSquareError4 + (true_prediction4 - actual_prediction4)**2

    average = average/(137*4)
    if average>prev_validate:
        learning_rate = learning_rate * 0.8

    prev_validate = average

    sumSquareError1 = np.sqrt(sumSquareError1/len(validateList))
    finalValue1 = np.sum(sumSquareError1)/10000

    sumSquareError2 = np.sqrt(sumSquareError2/len(validateList))
    finalValue2 = np.sum(sumSquareError2)/10000

    sumSquareError3 = np.sqrt(sumSquareError3/len(validateList))
    finalValue3 = np.sum(sumSquareError3)/10000

    sumSquareError4 = np.sqrt(sumSquareError4/len(validateList))
    finalValue4 = np.sum(sumSquareError4)/10000

    f2 = open("validate21.txt", "a")
    f2.write(str(finalValue1) + "\n")
    f2.write(str(np.min(sumSquareError1)) + "\n")
    f2.write(str(np.max(sumSquareError1)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate22.txt", "a")
    f2.write(str(finalValue2) + "\n")
    f2.write(str(np.min(sumSquareError2)) + "\n")
    f2.write(str(np.max(sumSquareError2)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate23.txt", "a")
    f2.write(str(finalValue3) + "\n")
    f2.write(str(np.min(sumSquareError3)) + "\n")
    f2.write(str(np.max(sumSquareError3)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate24.txt", "a")
    f2.write(str(finalValue4) + "\n")
    f2.write(str(np.min(sumSquareError4)) + "\n")
    f2.write(str(np.max(sumSquareError4)) + "\n")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate1.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate2.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate3.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

    f2 = open("validate4.txt", "a")
    f2.write("-----------------------------------------------END OF EPOCH-------------------------------------------")
    f2.write("\n")

loadData()
