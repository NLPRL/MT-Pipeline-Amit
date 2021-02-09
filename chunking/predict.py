from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import os
from collections import Counter
from keras.models import *
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dropout, Dense,concatenate
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.datasets import conll2000
# from keras_self_attention import SeqSelfAttention
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import numpy as np
max_len_char=18
max_len=116
features=5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR += '/'
chunking_file_path = BASE_DIR+'input.txt'
output_file = BASE_DIR+'output.txt'
dict_file = BASE_DIR+'vocabs.txt'
file_train=[]
def load_data(chunking_file_path, min_freq=1):
    # loading the vocabulary 
    global file_train
    file_train = _parse_data(open(chunking_file_path, encoding='utf-8'))
    word_counts = Counter(row[0].lower() for sample in file_train for row in sample)
    f=open(dict_file,'r', encoding='utf-8')
    vocab=f.readline()
    vocab.rstrip('\n')
    vocab=vocab.split(" ")
    characters=f.readline()
    characters.rstrip('\n')
    characters=characters.split(" ")
    pos_tags=f.readline()
    pos_tags.rstrip('\n')
    pos_tags=pos_tags.split(" ")
    class_labels=f.readline()
    class_labels.rstrip('\n')
    class_labels=class_labels.split(" ")
    features=f.readline()
    features.rstrip('\n')
    features=features.split(" ")
    f.close()
    global max_len_char
    for sample in file_train:
        for row in sample:
            max_len_char=max(max_len_char,len(row[0]))
    train = _process_data(file_train, vocab, pos_tags,characters,features)
    return train, (vocab, pos_tags, class_labels,characters,features)
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
def _process_data(data, vocab, pos_tags, characters, features, onehot=False):
    global max_len
    if max_len is None:
        max_len = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    f2idx = dict((w, i) for i,w in enumerate(features))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
    y_pos = [[pos_tags.index(w[1]) for w in s] for s in data]
    y_feature =[ [ [f2idx.get(i,1) for i in w[2:7]] for w in s] for s in data]
    defaultvalue=[0 for i in range(max_len_char)]
    x = pad_sequences(x, max_len)
    y_pos = pad_sequences(y_pos, max_len, value=0)
    y_feature = pad_sequences(y_feature,max_len, value=[0,0,0,0,0])
    if onehot:
        y_pos = numpy.eye(len(pos_tags), dtype='float32')[y]
    else:
        y_pos = numpy.expand_dims(y_pos, 2)
    return x, y_pos, y_feature
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
def main_chunker():
    out=[]
    train, voc = load_data(chunking_file_path)
    (X_test,x_pos_test,x_feature_test) = train
    (vocab, pos_tags, class_labels,characters,feat) = voc
    X_char_test=tocharacter(characters,vocab,X_test)
    json_file = open(BASE_DIR+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,custom_objects={'CRF': CRF})
    # load weights into new model
    model.load_weights(BASE_DIR+"model.h5")
    l4=[0]*max_len_char
    l4=numpy.asarray(l4)
    y= model.predict([X_test,np.array(X_char_test).reshape((len(X_char_test),max_len, max_len_char)),x_pos_test.reshape(len(x_pos_test),max_len),x_feature_test.reshape(len(x_feature_test),max_len,features)])
    y= y.argmax(-1)
    y_pred=[]
    #removing padding from the validation
    for i in range(len(X_test)):
        l=[]
        l1=[]
        for j in range(len(X_test[i])):
            if (X_test[i][j]==0):
                continue
            l.append(y[i][j])
        y_pred.append(l)
    #converting to numpy array
    test_y_pred=numpy.asarray(y_pred)
    #writing output for validation
    f1=open(output_file,'w', encoding='utf-8')
    for i in range(len(test_y_pred)):
        l1=[]
        for j in range(len(test_y_pred[i])):
            s="\t".join(file_train[i][j])+"\t"
            if (class_labels[test_y_pred[i][j]][0]=='O'):
                l1.append('O')
                s=s+'O'
            else:
                l1.append(class_labels[test_y_pred[i][j]])
                s=s+class_labels[test_y_pred[i][j]]
            f1.write(s+'\n')
        out.append(l1)
        f1.write('\n')
    f1.flush()
    f1.close()
    return out