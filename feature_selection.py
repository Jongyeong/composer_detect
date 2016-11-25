import json
import numpy as np
import scipy.stats as st

# define list (the number of class)
x1 = []
x2 = []
x3 = []

with open('train_data_3_thr0.9_normal', 'r') as trd:
    for tr_data in trd:
        x = json.loads(tr_data)
        if x[1][0] == 1:
            x1.append(x[0])
        elif x[1][1] == 1:
            x2.append(x[0])
        else:
            x3.append(x[0])

feature_num = len(x1[0])


# just change this value
x = x1
y = x2
# ------------------------ #

N_1 = len(x)
N_2 = len(y)

x_trans = np.array(x, ndmin=2).T
y_trans = np.array(y, ndmin=2).T

out = [0]*15

# output 1 = meaningful / 0 = almost meaningless
for n in range(feature_num):
    x_1 = np.mean(x_trans[n])
    x_2 = np.mean(y_trans[n])
    z = x_1-x_2
    sig_x1 = np.var(x[n])*(N_1-1)
    sig_x2 = np.var(y[n])*(N_2-1)
    sigma = sig_x1+sig_x2
    sigma /= N_1 + N_2 - 2
    N = (N_1 + N_2)/2
    Nn = (N_1+N_2)/float((N_1*N_2))
    confidence = st.t.interval(0.90, N_1 + N_2 - 2)[1]
    q = z/(np.sqrt(sigma*Nn))
    print confidence, q
    if q >= confidence:
        out[n] = 1
print out


