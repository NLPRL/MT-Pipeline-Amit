from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
from collections import Counter
from keras.models import *
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dropout, Dense,concatenate
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.datasets import conll2000
import os
# from keras_self_attention import SeqSelfAttention
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import model_from_json
from keras.models import load_model
import json
from sklearn.model_selection import train_test_split
import numpy as np
EPOCHS = 2
EMBED_DIM = 500
BiRNN_UNITS = 500
max_len_char=18
features=5
max_len=116
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR += '/'
pos_file_path = BASE_DIR+'sentinput.txt'
output_file = BASE_DIR+'final_output2.txt'
def load_data(pos_file_path, min_freq=1):

    voc = _parse_data(open(BASE_DIR+'vocabulary.txt', encoding='utf-8'))
    file_train = _parse_data(open(pos_file_path, encoding='utf-8'))
    word_counts = Counter(row[0].lower() for sample in file_train for row in sample)
    vocab = ['<pad>', '<unk>']
    vocab += [w for w, f in iter(word_counts.items()) if f >= min_freq]
    # print(len(voc[0]))
    vocab = voc[0][0]
    characters = voc[0][1]
    pos_tags = voc[0][2]
    feat = voc[0][3]
    # print(voc[0][3])
    global max_len_char
    for sample in file_train:
        for row in sample:
            max_len_char=max(max_len_char,len(row[0]))
    train = _process_data(file_train, vocab,characters,feat)
    return train, (vocab,pos_tags, characters,features)
def _parse_data(fh):
    string = fh.read()
    # print(string)
    data = []
    for sample in string.strip().split('\n\n'):
        data.append([row.split() for row in sample.split('\n')])
    fh.close()
    return data
def pad_words(l,max_len_char):
    length=len(l)
    l1=[0 for i in range(max_len_char-length)]
    l=l1+l
    return l
def _process_data(data, vocab, characters, features,onehot=False):
    global max_len
    if max_len is None:
        max_len = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    # print(features)
    f2idx = dict((w, i) for i,w in enumerate(features))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
    y_feature =[ [ [f2idx.get(i,1) for i in w[1:6]] for w in s] for s in data]
    defaultvalue=[0 for i in range(max_len_char)]
    x = pad_sequences(x, max_len)
    y_feature = pad_sequences(y_feature,max_len, value=[0,0,0,0,0])
    return x, y_feature
def tocharacter(characters,vocab,X_train):
    ''' Function to create word embedding into character embedding'''
    char2idx= dict((w,i) for i,w in enumerate(characters))
    idx2word= dict((i, w) for i, w in enumerate(vocab)) 
    l=[]
    for s in X_train:
        l1=[]
        for w in s:
            if (idx2word[w]=='<pad>'):
                l1.append([0]*max_len_char)
                continue
            if (idx2word[w]=='<unk>'):
                l1.append([1]*max_len_char)
                continue
            l2=[]
            for c in idx2word[w]:
                l2.append(char2idx.get(c,1))
            l2=pad_words(l2,max_len_char)
            l1.append(l2)
        l.append(l1)
    return numpy.asarray(l)
def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics,
    reports per classs recall, precision and F1 score'''

    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    'recall',
                                                    'precision',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')


def pos_main():
    train, voc = load_data(pos_file_path)
    (X,x_feature) = train
    (vocab, class_labels,characters,feat) = voc
    X_char=tocharacter(characters,vocab,X)
    X_test=X
    X_char_test=X_char
    x_feature_test=x_feature

    # Loading Model Weights
    json_file = open(BASE_DIR+'model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json,custom_objects={'CRF':CRF})
    loaded_model.load_weights(BASE_DIR+"model.h5")
    loaded_model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])

    # Prediction
    l4=[0]*max_len_char
    l4=numpy.asarray(l4)
    y= loaded_model.predict([X_test,np.array(X_char_test).reshape((len(X_char_test),max_len, max_len_char)),x_feature_test.reshape(len(x_feature_test),max_len,features)])
    y= y.argmax(-1)
    l4=[0]*max_len_char
    l4=numpy.asarray(l4)
    y_pred=[]
    ctest_y_pred=[]

    #removing padding from the modeloutput
    for i in range(len(X_test)):
        l=[]
        l2=[]
        l1=[]
        for j in range(len(X_test[i])):
            if (X_test[i][j]==0):
                continue
            l.append(y[i][j])
            ctest_y_pred.append(y[i][j])
        y_pred.append(l)

    #converting to numpy array
    test_y_pred=numpy.asarray(y_pred)

    #writing POS_Tags in output file
    f1=open(output_file,'w')
    out = []
    for i in range(len(test_y_pred)):
        out2 = []
        for j in range(len(test_y_pred[i])):
            out2.append(class_labels[test_y_pred[i][j]])
            s=str(class_labels[test_y_pred[i][j]])
            f1.write(s+'\n')
        out.append(out2)
        f1.write('\n')
    f1.flush()
    f1.close()
    return out