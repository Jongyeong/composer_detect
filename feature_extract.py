from music21 import *
import numpy as np
import json

# ------parameter------- #
threshold = 0.9 # distinguish measure
# ---------------------- #


def set_train_x(a, num):
    train_x_y = []
    if a == 0:
        tp = 'bwv'
        k = 0
    elif a == 1:
        tp = 'test'
        k = 1
    for i in range(num):
        try:
            scr = corpus.parse('train/' + tp + str(i))
            chrd = scr.chordify()
            mchrd = chrd.getElementsByClass('Measure')
            msr = []
            key_d = []
            tempo_d = []
            mchrd_l = []
            train_loop(mchrd, mchrd_l, msr, key_d, tempo_d)
            d_msr = distinct_m(msr)
            data_set(d_msr, mchrd_l, i, k, train_x_y)
        except Exception as e:
            print(tp + str(i), e)
    return train_x_y


# derive properties
def train_loop(mchrd, mchrd_l, msr, key_d, tempo_d):
    for j in range(0, len(mchrd), 2):
        try:
            ad_msr = []
            ad_mchrd_l = []
            for k in range(len(mchrd[j])):
                if type(mchrd[j][k]) is chord.Chord:
                    ad_mchrd_l.append(mchrd[j][k])
                    ad_msr += mchrd[j][k].normalOrder
                elif type(mchrd[j][k]) is key.Key:
                    key_d.append(mchrd[j][k])
                elif type(mchrd[j][k]) is meter.TimeSignature:
                    tempo_d.append(mchrd[j][k])
                elif type(mchrd[j][k]) is bar.Barline:
                    if mchrd[j][k].style == 'final':
                        mchrd_l.append(ad_mchrd_l)
                        return

            for k in range(len(mchrd[j + 1])):
                if type(mchrd[j + 1][k]) is chord.Chord:
                    ad_mchrd_l.append(mchrd[j + 1][k])
                    ad_msr += mchrd[j + 1][k].normalOrder
                elif type(mchrd[j + 1][k]) is key.Key:
                    key_d.append(mchrd[j + 1][k])
                elif type(mchrd[j + 1][k]) is meter.TimeSignature:
                    tempo_d.append(mchrd[j + 1][k])
                elif type(mchrd[j + 1][k]) is bar.Barline:
                    if mchrd[j + 1][k].style == 'final':
                        msr.append(ad_msr)
                        mchrd_l.append(ad_mchrd_l)
                        return
            mchrd_l.append(ad_mchrd_l)
            msr.append(ad_msr)
        except Exception as e:
            print e
            break
    return


# distinct measure detection
def distinct_m(msr):
    cor = [msr[0]]
    result = [0]
    for i in range(1, len(msr)):
        cnt = 0
        for x in cor:
            if np.correlate(msr[i], msr[i])[0]*threshold < np.correlate(msr[i], x)[0]:
                cnt = 1
                break
            elif np.correlate(msr[i], msr[i])[0] == 0:
                cnt = 1
                break
        if cnt == 0:
            cor.append(msr[i])
            result.append(i)
    return result


def data_set(d_msr, mchrd_l, num, k, train_x_y):
    for i in d_msr:
        a = np.around(count(mchrd_l[i]) + mean_stdvar(mchrd_l[i]) + itvec(mchrd_l[i]), decimals=6).tolist()
        if np.isnan(a).any():
            return
        b = data_output_set(num, k)
        ad_train_x_y = np.array([a, b])
        train_x_y.append(ad_train_x_y)
    return


def data_output_set(i, n):
    d_out = [0, 0, 0, 0]
    if n == 0:
        if i < 82:
            d_out[0] = 1
        elif i < 162:
            d_out[1] = 1
        elif i < 208:
            d_out[2] = 1
        else:
            d_out[3] = 1
    elif n == 1:
        if i < 20:
            d_out[0] = 1
        elif i < 39:
            d_out[1] = 1
        elif i < 63:
            d_out[2] = [1]
        else:
            d_out[3] = 1
    return d_out


def count(mchrd_l):
    output = [0.0, 0.0, 0.0, 0.0, 0.0]
    if len(mchrd_l) == 0:
        return output
    de = 1.0/len(mchrd_l)
    for c in mchrd_l:
        if c.hasZRelation:
            output[0] += de
        if c.canBeDominantV():
            output[1] += de
        if c.canBeTonic():
            output[2] += de
        if c.hasAnyRepeatedDiatonicNote():
            output[3] += de
        if c.isConsonant():
            output[4] += de
    return output


def mean_stdvar(mchrd_l):
    ad_hoc_f = []
    ad_hoc_m = []
    for c in mchrd_l:
        ad_hoc_f.append(c.forteClassNumber)
        ad_hoc_m.append(c.multisetCardinality)

    return [np.mean(ad_hoc_f)/10., np.std(ad_hoc_f)/np.sqrt(10), np.mean(ad_hoc_m)/10., np.std(ad_hoc_m)/np.sqrt(10)]


def itvec(mchrd_l):
    output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for c in mchrd_l:
        output = np.add(output, c.intervalVector)

    return list(np.divide(output, len(mchrd_l)))


# make train_data list as file
with open('train_data_4_thr0.9_normal', 'w') as trd:
    for tr_data in set_train_x(0, 303):
        trd.write("{}\n".format(json.dumps(tr_data.tolist())))


# make test_data list as file
with open('test_data_4_thr0.9_normal', 'w') as tsd:
    for ts_data in set_train_x(1, 91):
        tsd.write("{}\n".format(json.dumps(ts_data.tolist())))
