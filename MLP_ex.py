import json
import numpy as np
import random
import pandas as pd

# ------parameter------- #
input_dim = 15
output_dim = 2
hidden_dim = 120
layer_num = 4   # includes input layer


init = [0.2, 0.4]# [ init_weight_std, init_bias_std]
training_factor = [2, 2210]#[ total_epoch , test_period ]

lf_step = 0.03
lfl = np.arange(0.01, 0.301, 0.001)
# ---------------------- #

num_neuron = [input_dim] + (layer_num-2)*[hidden_dim] + [output_dim]
train_data = []
test_data = []

with open('train_data_round', 'r') as trd:
    for tr_data in trd:
        train_data.append(json.loads(tr_data))

with open('test_data_round', 'r') as tsd:
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
    return 1.0 / (1 + np.exp(-2*y+1))


def back_prop(y_hat, weight, d_output):
    # r = L
    net_delta = [d_output-y_hat[-1]]
    for r in range(layer_num-2, -1, -1):
        e = y_hat[r]
        sm = (net_delta[0].T.dot(weight[r+1])).T
        net_delta.insert(0, 2*e*(1-e)*sm)
    return net_delta


def update(y_hat, weight, bias, net_delta, epoch):
    for r in range(1, layer_num):
        del_w = (lft * y_hat[r - 1].dot(net_delta[r].T)).T
        del_b = lft * net_delta[r]
        weight[r] = np.add(weight[r], del_w)
        bias[r] = np.add(bias[r], del_b)
    return [y_hat, weight, bias]


def adjust_lf(y_hat, data_output, cost, reg):
    cost_n = crossentropy(y_hat[-1], data_output)
    if cost == 0.0:
        return [cost_n, reg]
    if np.isnan(cost_n):
        cost_n = np.nan_to_num(cost_n)
        print 'nan loss', cost_n,'\n', y_hat, d_output
    loss = cost_n/cost
    step = int((loss - 1.0)/lf_step)
    if step >= 1:
        reg -= int(np.log2(step))    # decrease learning factor
        if reg < 0:
            reg = 0
    elif step <= -1:
        step = -step
        reg += int(np.log2(step))    # increase learning factor
        if reg > len(lfl)-1:
            reg = len(lfl)-1

    return [cost_n, reg]


def crossentropy(y_hat, y):
    cost = 0
    for i in range(len(y)):
        if y_hat[i] == 0:
            break
        cost -= y[i]*np.log2(y_hat[i])
    return cost


[y_hat, weight, bias] = initialize_net()
cost = 0.0
reg = len(lfl)*2/3
lft = lfl[reg]
min_error = 1
for epoch in range(training_factor[0]):
    random.shuffle(train_data)
    random.shuffle(test_data)
    for index in range(len(train_data)):
        d_input = np.array(train_data[index][0], ndmin=2).T
        d_output = np.array(train_data[index][1], ndmin=2).T
        y_hat = feed_forward(d_input, y_hat, weight, bias)
        net_delta = back_prop(y_hat, weight, d_output)
        [y_hat, weight, bias] = update(y_hat, weight, bias, net_delta, epoch)
        [cost, reg] = adjust_lf(y_hat, d_output, cost, reg)
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
            if test_error <= min_error:
                min_error = test_error
                w = weight
                b = bias
            print('epoch : ', epoch, 'test error : ', test_error)
            print lft
print('min error rate: ', min_error)
