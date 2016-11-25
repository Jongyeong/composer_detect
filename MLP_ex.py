import json
import numpy as np
import random
import pandas as pd

# ------parameter------- #
input_dim = 15
output_dim = 3
hidden_dim = 80
layer_num = 7

num_neuron = [input_dim] + (layer_num-2)*[hidden_dim] + [output_dim]

init = [0.1, 0.2]# [ init_weight_std, init_bias_std]
training_factor = [500, 4763]#[ total_epoch , test_period ]

reg_lf = 1.05
lfl = np.arange(0.4, 0, -0.01)
# ---------------------- #


train_data = []
test_data = []

with open('train_data_3_thr0.9_normal', 'r') as trd:
    for tr_data in trd:
        train_data.append(json.loads(tr_data))

with open('test_data_3_thr0.9_normal', 'r') as tsd:
    for ts_data in tsd:
        test_data.append(json.loads(ts_data))


def initialize_net():
    y_hat = []
    weight = [[0]]
    bias = [[0]]
    for r in range(layer_num):  # y_hat : kr * 1
        y_hat.append(np.zeros((num_neuron[r], 1)))

    for r in range(1, layer_num):  # weight : kr * kr-1 matrix / bias : kr * 1
        weight.append(np.multiply(init[0], np.random.randn(num_neuron[r], num_neuron[r-1])))
        bias.append(np.multiply(init[1], np.random.randn(num_neuron[r], 1)))

    return [y_hat, weight, bias]


def feed_forward(d_input, y_hat, weight, bias):
    y_hat[0] = d_input
    for r in range(1, layer_num):
        if r == layer_num-1:
            y_hat[r] = softmax(np.dot(weight[r], y_hat[r-1]) + bias[r])
        else:
            y_hat[r] = logistic(np.dot(weight[r], y_hat[r-1]) + bias[r])
    return y_hat


def softmax(y):
    e = np.exp(y)
    dist = np.divide(e, np.sum(e))
    return dist


def logistic(y):
    return 1.0 / (1 + np.exp(-y))


def back_prop(y_hat, weight, d_output):
    # r = L
    net_delta = [d_output-y_hat[-1]]
    for r in range(layer_num-2, -1, -1):
        e = y_hat[r]
        sm = (net_delta[0].T.dot(weight[r+1])).T
        net_delta.insert(0, (e*(1-e)*sm))
    return net_delta


def update(y_hat, weight, bias, net_delta, epoch):
    for r in range(1, layer_num):
        del_w = (lft * y_hat[r - 1].dot(net_delta[r].T)).T
        del_b = lft * net_delta[r]
        weight[r] = np.add(weight[r], del_w)
        bias[r] = np.add(bias[r], del_b)
    return [y_hat, weight, bias]


def adjust_lf(y_hat, data_output, loss, reg):
    loss_n = crossentropy(y_hat[-1], data_output)
    if np.isnan(loss_n):
        loss_n = np.nan_to_num(loss_n)
        print 'nan loss', loss_n
    if loss_n > loss:
        if reg < len(lfl)-1:
            reg += 1    # decrease learning factor
    elif loss_n*reg_lf < loss:
        if reg != 0:
            reg -= 1    # increase learning factor
    return [loss_n, reg]


def crossentropy(y_hat, y):
    cost = 0
    for i in range(len(y)):
        if y_hat[i] == 0:
            break
        cost -= y[i]*np.log2(y_hat[i])
    return cost


[y_hat, weight, bias] = initialize_net()
loss = 0.0
reg = len(lfl)*1/3
lft = lfl[reg]

for epoch in range(training_factor[0]):
    random.shuffle(train_data)
    random.shuffle(test_data)
    for index in range(len(train_data)):
        d_input = np.array(train_data[index][0], ndmin=2).T
        d_output = np.array(train_data[index][1], ndmin=2).T
        y_hat = feed_forward(d_input, y_hat, weight, bias)
        net_delta = back_prop(y_hat, weight, d_output)
        [y_hat, weight, bias] = update(y_hat, weight, bias, net_delta, epoch)
        [loss, reg] = adjust_lf(y_hat, d_output, loss, reg)
        lft = lfl[reg]
        if pd.isnull(y_hat).any():
            print 'nan'
        if index == training_factor[1]-1:
            test_error = 0.0
            for n in range(len(test_data)):
                test_input = np.array(test_data[n][0], ndmin=2).T
                test_output = np.array(test_data[n][1], ndmin=2).T
                x = feed_forward(test_input, y_hat, weight, bias)
                if np.argmax(x[-1]) != np.argmax(test_output):
                    test_error += 1
            test_error /= len(test_data)
            print('epoch : ', epoch, 'test error : ', test_error)
            print lft
