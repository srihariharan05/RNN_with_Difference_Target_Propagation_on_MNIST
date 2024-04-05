import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
import sys
import scipy.io
import random
from itertools import permutations

def tanh_activation(Z):
     return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)-np.exp(-Z)) # this is the tanh function can also be written as np.tanh(Z)

def softmax_activation(Z):
        e_x = np.exp(Z - np.max(Z))  # this is the code for softmax function 
        return e_x / e_x.sum(axis=0) 

def delta_cross_entropy(predicted_output,original_t_output):
    li = []
    grad = predicted_output
    for i,l in enumerate(original_t_output): #check if the value in the index is 1 or not, if yes then take the same index value from the predicted_ouput list and subtract 1 from it. 
        if l == 1:
    #grad = np.asarray(np.concatenate( grad, axis=0 ))
            grad[i] -= 1
    return grad

def tanh_activation_backward(x,top_diff):
    output = np.tanh(x)
    return (1.0 - np.square(output)) * top_diff

def multiplication_backward(weights,x,dz):
    gradient_weight = np.array(np.dot(np.asmatrix(dz),np.transpose(np.asmatrix(x))))
    chain_gradient = np.dot(np.transpose(weights),dz)
    return gradient_weight,chain_gradient
def add_backward(x1,x2,dz):
    dx1 = dz * np.ones_like(x1)
    dx2 = dz * np.ones_like(x2)
    return dx1,dx2


def Rnn_forward(input, input_weights, activation_weights, prev_memory):
    forward_params = []
    U_frd = np.dot(activation_weights,prev_memory)
    W_frd = np.asarray(np.dot(input_weights,input))
    sum_s = W_frd + U_frd
    ht_activated = tanh_activation(sum_s)
    #yt_unactivated = np.asarray(np.dot(output_weights,  ht_activated))
    #yt_activated = softmax_activation(yt_unactivated)
    #yt_unactivated = np.asarray(np.zeros(output_dim,1))
    forward_params.append([W_frd,U_frd,sum_s])
    return ht_activated,forward_params

def full_forward_prop(T, input_seq ,input_weights,activation_weights,prev_memory,output_weights):
    predicted_output = []
    memory = {}
    #should we give prev_memory to a new Forward prop?
    prev_ht_activation = prev_memory
    for t in range(0,T):
        curr_activation, params = Rnn_forward(input_seq[t], input_weights, activation_weights, prev_ht_activation)
        prev_ht_activation = curr_activation
        memory["ht" + str(t)] = prev_ht_activation
        memory["params" + str(t)] = params
    yt_unactivated = np.asarray(np.dot(output_weights,  curr_activation))
    predicted_output = softmax_activation(yt_unactivated)
    return predicted_output, memory 

def calculate_loss(expected_output,predicted_output):
    loss = -sum(expected_output[i]*np.log2(predicted_output[i]) for i in range(len(expected_output)))
    # should we average it by 10?
    return loss
    """ total_loss = 0
    layer_loss = []
    for y,y_ in zip(output_mapper.values(),predicted_output): # this for loop calculation is for the first equation, where loss for each time-stamp is calculated
        loss = -sum(y[i]*np.log2(y_[i]) for i in range(len(y)))
        loss = loss/ float(len(y))
        layer_loss.append(loss) 
    for i in range(len(layer_loss)): #this the total loss calculated for all the time-stamps considered together. 
        total_loss  = total_loss + layer_loss[i]
    return total_loss/float(len(predicted_output)) """

def single_backprop(X,input_weights,activation_weights,output_weights,ht_activated,dLo,forward_params_t,diff_s,prev_s):# inlide all the param values for all the data thats there
    W_frd = forward_params_t[0][0] 
    U_frd = forward_params_t[0][1]
    ht_unactivated = forward_params_t[0][2]
    #yt_unactivated = forward_params_t[0][3]
    dV,dsv = multiplication_backward(output_weights,ht_activated,dLo)
    ds = np.add(dsv,diff_s) # used for truncation of memory 
    dadd = tanh_activation_backward(ht_unactivated, ds)
    dmulw,dmulu = add_backward(U_frd,W_frd,dadd)
    dW, dprev_s = multiplication_backward(activation_weights, prev_s ,dmulw)
    dU, dx = multiplication_backward(input_weights, X, dmulu) #input weights
    return (dprev_s, dU, dW, dV)

def rnn_backprop(input_seq,predicted_output,memory,expected_output,dU,dV,dW,input_weights,output_weights,activation_weights, sequence_lenth):
    # we start the backprop from the last timestamp. 
    t = sequence_lenth-1
    prev_s_t = memory["ht" + str(t-1)] #p.zeros((hidden_dim,1)) #required as the first timestamp does not have a previous activation or memory
    diff_s = np.zeros((hidden_dim,1)) # this is used for the truncating purpose of restoring a previous information from the before level
    ht_activated = memory["ht" + str(t)]
    forward_params_t = memory["params"+ str(t)] 
    dLo = delta_cross_entropy(predicted_output,expected_output) #the loss derivative for that timestamp
    dprev_s, dU_t, dW_t, dV_t = single_backprop(input_seq[t],input_weights,activation_weights,output_weights,ht_activated,dLo,forward_params_t,diff_s,prev_s_t)
    #prev = t-1
    dLo = np.zeros((output_dim,1)) #here the loss deriative is turned to 0 as we do not require it for the turncated information.
    # the following code is for the trunated bptt and its for each time-stamp. 
    for i in range(t-1,-1,-1):
        forward_params_t = memory["params" + str(i)]
        ht_activated = memory["ht" + str(i)]
        prev_s_t = np.zeros((hidden_dim,1)) if i == 0 else memory["ht" + str(i-1)]
        dprev_s, dU_i, dW_i, dV_i = single_backprop(input_seq[i] ,input_weights,activation_weights,output_weights,ht_activated,dLo,forward_params_t,dprev_s,prev_s_t)
        dU_t += dU_i #adding the previous gradients on lookback to the current time sequence 
        dW_t += dW_i
    dV += dV_t 
    dU += dU_t
    dW += dW_t
    return (dU, dW, dV)

def sgd_step(learning_rate, dU,dW,dV, input_weights, activation_weights,output_weights ):
    input_weights -= learning_rate* dU
    activation_weights -= learning_rate * dW
    output_weights -=learning_rate * dV
    return input_weights,activation_weights,output_weights


learning_rate = 0.0001    
nepoch = 5               
batch_size = 100;
training_size = 60000
batch_number = training_size//batch_size
rand_list =[i for i in range (training_size)]
random.shuffle(rand_list)
print(rand_list)
input_dim = 1 # 128 columns in each row of an image
T = 64 # 128 rows   # length of sequence
outlook = 1
hidden_dim = 100 # as mentioend in OG repo       
output_dim = 10 # this is the total unique words in the vocabulary
bptt_truncate = 2 # here instead of using vanilla back-propogation, we use a truncated back propogation, so while doing BP we look back at atmost 2 cells 

input_weights = np.random.uniform(0, 1, (hidden_dim,input_dim))
activation_weights = np.random.uniform(0,1, (hidden_dim, hidden_dim))
output_weights = np.random.uniform(0,1, (output_dim,hidden_dim))
prev_memory =  np.random.uniform(0,1, (hidden_dim,1))

dU = np.zeros(input_weights.shape)
dV = np.zeros(output_weights.shape)
dW = np.zeros(activation_weights.shape)

data_set = scipy.io.loadmat('MNIST_TrainSet_0to1_8x8pixel.mat')     #MNIST dataset in 8x8 resolution
data_set = data_set['number']
label_set = scipy.io.loadmat('MNIST_TrainSet_Label.mat')
label = label_set['label']
#for i in range (60000):
#    print(label[0][i])
identity_matrix = np.identity(output_dim, dtype = float)

print( "starting RNN training:\n batch_number = {} \n ".format(batch_number))
input_seq =[];
expected_output=[]
losses = []
random.seed()       # to generate random number
for epoch in range(nepoch):
    for batch in range(batch_number):
        for i in range(batch*batch_size, (batch+1)*batch_size):
            inst = rand_list[i]
            input_seq = data_set[:,inst]
            label_value = label[0][inst]
            expected_output = identity_matrix[label_value,:]
            #print("iteration: {} \tlabel : {} \n input seq  {} \n expected output : {}".format(i,label_value,input_seq,expected_output))
            predicted_output,memory = full_forward_prop(T, input_seq ,input_weights,activation_weights,prev_memory,output_weights)
            dU,dW,dV = rnn_backprop(input_seq,predicted_output,memory,expected_output,dU,dV,dW,input_weights,output_weights,activation_weights,T)
        input_weights,activation_weights,output_weights= sgd_step(learning_rate,dU,dW,dV,input_weights,activation_weights,output_weights)
    rand_n = random.randint(0,60000)
    input_seq = data_set[:,rand_n]
    label_value = label[0][rand_n]
    expected_output = identity_matrix[label_value,:]
    predicted_output,memory = full_forward_prop(T, input_seq ,input_weights,activation_weights,prev_memory,output_weights)    
    loss = calculate_loss(expected_output, predicted_output)
    losses.append(loss)
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Loss after  epoch=%d: %f" % (time,epoch, loss))
    print("label : {} \n input seq  {} \n expected output : {}".format(label_value,input_seq,expected_output))
    sys.stdout.flush()        
    

