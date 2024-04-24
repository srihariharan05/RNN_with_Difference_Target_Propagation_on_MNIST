import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
#import sys
import scipy.io
import random
from itertools import permutations

def tanh_activation(Z):
    return np.tanh(Z)
     #return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)-np.exp(-Z)) # this is the tanh function can also be written as np.tanh(Z)

def softmax_activation(Z):
        e_x = np.exp(Z - np.max(Z))  # this is the code for softmax function 
        return e_x / e_x.sum(axis=0) 

def delta_cross_entropy(unactivated_output,original_t_output):
    li = []
    grad = unactivated_output
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


def Rnn_forward(input, input_weights, activation_weights, prev_memory, hid_bias):
    forward_params = []
    W_frd = np.dot(activation_weights,prev_memory)
    U_frd = np.asarray(np.dot(input_weights,input))
    sum_s = W_frd + U_frd + hid_bias
    ht_activated = tanh_activation(sum_s)
    #yt_unactivated = np.asarray(np.dot(output_weights,  ht_activated))
    #yt_activated = softmax_activation(yt_unactivated)
    #yt_unactivated = np.asarray(np.zeros(output_dim,1))
    forward_params.append([W_frd,U_frd,sum_s])
    return ht_activated,forward_params

def full_forward_prop(T, input_seq ,input_weights,activation_weights,prev_memory,output_weights, hid_bias, out_bias):
    #predicted_output = []
    predicted_output = np.zeros((output_dim,1))
    memory = {}
    #should we give prev_memory to a new Forward prop?
    prev_ht_activation = prev_memory
    for t in range(0,T):
        curr_activation, params = Rnn_forward(input_seq[t], input_weights, activation_weights, prev_ht_activation,hid_bias)
        prev_ht_activation = curr_activation
        memory["ht" + str(t)] = prev_ht_activation
        memory["params" + str(t)] = params
    yt_unactivated = np.asarray(np.dot(output_weights,  curr_activation)) + out_bias
    predicted_output = softmax_activation(yt_unactivated)
    return predicted_output, yt_unactivated,memory 

def calculate_loss(expected_output,predicted_output):
    loss = -sum(expected_output[i]*np.log2(predicted_output[i]) for i in range(len(expected_output)))
    # should we average it by 10? - yes
    loss /= len(expected_output)
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
    db_out = dLo 
    ds = np.add(dsv,diff_s) # used for truncation of memory 
    dadd = tanh_activation_backward(ht_unactivated, ds)
    db_hid = dadd
    dmulw,dmulu = add_backward(U_frd,W_frd,dadd)
    dW, dprev_s = multiplication_backward(activation_weights, prev_s ,dmulw)
    dU, dx = multiplication_backward(input_weights, X, dmulu) #input weights
    return (dprev_s, dU, dW, dV, db_out, db_hid)

def rnn_backprop(input_seq,predicted_output,memory,expected_output,dU,dV,dW,input_weights,output_weights,activation_weights, sequence_lenth,db_hid, db_out):
    # we start the backprop from the last timestamp. 
    t = sequence_lenth-1
    prev_s_t = memory["ht" + str(t-1)] #p.zeros((hidden_dim,1)) #required as the first timestamp does not have a previous activation or memory
    diff_s = np.zeros((hidden_dim,1)) # this is used for the truncating purpose of restoring a previous information from the before level
    ht_activated = memory["ht" + str(t)]
    forward_params_t = memory["params"+ str(t)] 
    dLo = delta_cross_entropy(predicted_output,expected_output) #the loss derivative for that timestamp
    dprev_s, dU_t, dW_t, dV_t,db_out_t, db_hid_t = single_backprop(input_seq[t],input_weights,activation_weights,output_weights,ht_activated,dLo,forward_params_t,diff_s,prev_s_t)
    #prev = t-1
    dLo = np.zeros((output_dim,1)) #here the loss deriative is turned to 0 as we do not require it for the turncated information.
    # the following code is for the trunated bptt and its for each time-stamp. 
    for i in range(t-1,-1,-1):
        forward_params_t = memory["params" + str(i)]
        ht_activated = memory["ht" + str(i)]
        prev_s_t = np.zeros((hidden_dim,1)) if i == 0 else memory["ht" + str(i-1)]
        dprev_s, dU_i, dW_i, dV_i, db_out_i, db_hid_i = single_backprop(input_seq[i] ,input_weights,activation_weights,output_weights,ht_activated,dLo,forward_params_t,dprev_s,prev_s_t)
        dU_t += dU_i #adding the previous gradients on lookback to the current time sequence 
        dW_t += dW_i
        db_hid_t += db_hid_i
    dV += dV_t 
    dU += dU_t
    dW += dW_t
    db_hid += db_hid_t
    db_out += db_out_t
    return (dU, dW, dV,db_hid,db_out)

def sgd_step(learning_rate, dU,dW,dV,db_hid,db_out, input_weights, activation_weights,output_weights, hid_bias,out_bias ):
    input_weights -= learning_rate* dU
    activation_weights -= learning_rate * dW
    output_weights -=learning_rate * dV
    out_bias -=learning_rate * db_out
    hid_bias  -=learning_rate * db_hid
    return input_weights,activation_weights,output_weights,out_bias,hid_bias

bypass_training =False;
row_wise_seq = False
learning_rate = 0.0001    
nepoch = 5               
batch_size = 100;
training_size =58000
test_size = 60000-training_size
batch_number = training_size//batch_size
rand_list =[i for i in range (60000)]
# 128 columns in each row of an image
if (row_wise_seq == False):
    input_dim = 1
    T =64  
else : 
    input_dim = 8
    T = 8 # 128 rows   # length of sequence
outlook = 1
hidden_dim = 100 #       
output_dim = 10 # this is the total unique words in the vocabulary
bptt_truncate = 2 # here instead of using vanilla back-propogation, we use a truncated back propogation, so while doing BP we look back at atmost 2 cells 

den = np.sqrt(hidden_dim)
input_weights = np.random.uniform(0, 1, (hidden_dim,input_dim))
input_weights /= den
activation_weights = np.random.uniform(0,1, (hidden_dim, hidden_dim))
activation_weights /= den
output_weights = np.random.uniform(0,1, (output_dim,hidden_dim))
output_weights /= den
prev_memory =  np.zeros((hidden_dim,1))
output_bias = np.zeros((output_dim,1))
hidden_bias = np.zeros((hidden_dim,1))

dU = np.zeros(input_weights.shape)
dV = np.zeros(output_weights.shape)
dW = np.zeros(activation_weights.shape)
db_output = np.zeros(output_bias.shape)
db_hidden = np.zeros(hidden_bias.shape)

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
correct_pred_training =0;
training_accuracy_list = [ 0 for x in range(nepoch)]

random.seed()       # to generate random number
random.shuffle(rand_list)
training_set = rand_list[0:training_size]
test_set = rand_list[training_size:]
input_row_seq =[]
if (bypass_training == False):
    for epoch in range(nepoch):
        correct_pred_training =0
        random.shuffle(training_set)        
        for batch in range(batch_number):
            #reset gradients 
            dU = np.zeros(input_weights.shape)
            dV = np.zeros(output_weights.shape)
            dW = np.zeros(activation_weights.shape)
            db_output = np.zeros(output_bias.shape)
            db_hidden = np.zeros(hidden_bias.shape)
            for i in range(batch*batch_size, (batch+1)*batch_size):
                inst = training_set[i]
                input_seq = data_set[:,inst]
                label_value = label[0][inst]
                expected_output = identity_matrix[label_value,:]
                if row_wise_seq == True:
                    for seq_i in range(T):
                        input_row_seq.append( input_seq[seq_i*input_dim : (seq_i +1)*input_dim])
                        print (input_row_seq)
                    predicted_output,unactivated_output,memory = full_forward_prop(T, input_row_seq ,input_weights,activation_weights,prev_memory,output_weights,hidden_bias,output_bias)
                    dU,dW,dV,db_hidden,db_output = rnn_backprop(input_row_seq,predicted_output,memory,expected_output,dU,dV,dW,input_weights,output_weights,activation_weights,T,db_hidden,db_output)
                    
                #print("iteration: {} \tlabel : {} \n input seq  {} \n expected output : {}".format(i,label_value,input_seq,expected_output))
                else:
                    predicted_output,unactivated_output,memory = full_forward_prop(T, input_seq ,input_weights,activation_weights,prev_memory,output_weights,hidden_bias,output_bias)                    
                    dU,dW,dV,db_hidden,db_output = rnn_backprop(input_seq,predicted_output,memory,expected_output,dU,dV,dW,input_weights,output_weights,activation_weights,T,db_hidden,db_output)
                np.clip(dV,-1,1,out=dV)
                np.clip(dW,-1,1,out=dW)
                np.clip(dU,-1,1,out=dU)
                np.clip(db_hidden,-1,1,out=db_hidden)
                np.clip(db_output,-1,1,out=db_output)
                max = predicted_output[0]
                k =0
                for j in predicted_output:
                    if (j >= max):
                        max = j
                        predicted_label = k
                    k = k+1
                if predicted_label == label_value:
                    correct_pred_training +=1
                print("gradient values duting batch : {}----------------------------------------------\n".format(batch))
                print ("\ndw:-----\n")
                print(str(dW))
                print("dU: --------\n")
                print(str(dU))
                print("dV: -----------\n")
                print(str(dV))
            dU /= batch_size
            dV /= batch_size
            dW /= batch_size
            db_hidden /= batch_size
            db_output /= batch_size    
            input_weights,activation_weights,output_weights,output_bias,hidden_bias= sgd_step(learning_rate,dU,dW,dV,db_hidden,db_output,input_weights,activation_weights,output_weights,hidden_bias,output_bias)
            #print("batch {} training accuracy = {} \n".format(batch,training_accuracy))
        training_accuracy = correct_pred_training/training_size
        training_accuracy_list[epoch] = training_accuracy
        rand_n = random.randint(0,training_size-1)
        input_seq = data_set[:,rand_n]
        label_value = label[0][rand_n]
        expected_output = identity_matrix[label_value,:]
        predicted_output,unactivated_output,memory = full_forward_prop(T, input_seq ,input_weights,activation_weights,prev_memory,output_weights,hidden_bias,output_bias)    
        loss = calculate_loss(expected_output, predicted_output)
        losses.append(loss)
        max = predicted_output[0]
        k =0
        for j in predicted_output:
            if (j >= max):
                max = j
                predicted_label = k
            k = k+1
        
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s: Loss after  epoch=%d: %f" % (time,epoch, loss))
        print("label : {} \n input seq  {} \n expected output : {} \n predicted output = {} \n predicted label = {} training accuracy : {}".format(label_value,input_seq,expected_output,predicted_output,predicted_label,training_accuracy))
        #sys.stdout.flush()        
    #plot accuracy graph
    plt.plot([ x for x in range(nepoch)],training_accuracy_list)
    plt.title("Training accuracy")
    #plt.show()
    plt.savefig('training_accuracy_graph.png', bbox_inches='tight')
    with open ("RNN_weights.npy",'wb') as f:
        np.save(f,input_weights)
        np.save(f,output_weights)
        np.save(f,activation_weights)
else:
    with open('RNN_weights.npy', 'rb') as f:
        input_weights = np.load(f)
        output_weights = np.load(f)
        activation_weights = np.load(f)


#testing
test_losses=[]
correct_predictions ={}
file = open( "log.txt",'w+')
for i in range(test_size):
    inst = test_set[i]
    input_seq = data_set[:,inst]
    label_value = label[0][inst]
    expected_output = identity_matrix[label_value,:]
    if row_wise_seq == True:
        for seq_i in range(T):
            input_row_seq[seq_i] = input_seq[seq_i*input_dim : (seq_i +1)*input_dim]
        predicted_output,unactivated_output,memory = full_forward_prop(T, input_row_seq ,input_weights,activation_weights,prev_memory,output_weights,hidden_bias,output_bias)
    else:
        predicted_output,unactivated_output,memory = full_forward_prop(T, input_seq ,input_weights,activation_weights,prev_memory,output_weights,hidden_bias,output_bias)                    
    max = predicted_output[0]
    test_loss = calculate_loss(expected_output, predicted_output)
    test_losses.append(test_loss)
    max = predicted_output[0]
    iter =0
    predicted_label =-1
    for j in predicted_output:
        if (j >= max):
            max = j
            predicted_label = iter
        iter = iter+1
    if predicted_label == label_value:
        correct_predictions[inst] = (predicted_label,predicted_output)
    file.write("label : {} \n input seq  {} \n expected output : {} \n predicted output = {} \n predicted label ={}".format(label_value,input_seq,expected_output,predicted_output,predicted_label))
test_accuracy = len(correct_predictions)/test_size
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
file.write("test Losses ")
for tst_loss in test_losses:
    file.write(str(tst_loss))
file.write("Test Accuracy = {}".format(test_accuracy))
file.close()
#sys.stdout.flush()   



