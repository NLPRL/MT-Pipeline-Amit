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
from keras_self_attention import SeqSelfAttention
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split
import numpy as np
EPOCHS = 5
EMBED_DIM = 500
BiRNN_UNITS = 500
max_len_char=-1
features=5
max_len=None
chunking_file_path = 'input_file.txt'
output_file = 'output.txt'
def load_data(chunking_file_path, min_freq=1):

    file_train = _parse_data(open(chunking_file_path))
    word_counts = Counter(row[0].lower() for sample in file_train for row in sample)
    vocab = ['<pad>', '<unk>']
    vocab += [w for w, f in iter(word_counts.items()) if f >= min_freq]
    pos_tags = sorted(list(set(row[1] for sample in file_train for row in sample)))
    characters = sorted(list(set(i for sample in file_train for row in sample for i in row[0])))
    features = sorted(list(set(i for sample in file_train for row in sample for i in row[2:7])))
    characters.insert(0,'<unk>')
    characters.insert(0,'<pad>')
    pos_tags.insert(0,'<unk>')
    pos_tags.insert(0,'<pad>')
    features.insert(0,'<pad>')
    global max_len_char
    for sample in file_train:
        for row in sample:
            max_len_char=max(max_len_char,len(row[0]))
    train = _process_data(file_train, vocab, pos_tags,characters,features)
    return train, (vocab, pos_tags,characters,features)
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
def _process_data(data, vocab, pos_tags, characters, features,onehot=False):
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

train, voc = load_data(chunking_file_path)
(X,y,x_feature) = train
(vocab, class_labels,characters,feat) = voc
X_char=tocharacter(characters,vocab,X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2018)
X_char_train, X_char_test, __ , __ = train_test_split(X_char, y, test_size=0.3,random_state=2018)
x_feature_train,x_feature_test,__, ___ = train_test_split(x_feature, y, test_size=0.3,random_state=2018)
print('==== training BiLSTM-CRF ====')
# word embedding
word_in=Input(shape=(X_train.shape[1],),name='word_in')
emb_word=(Embedding(len(vocab), EMBED_DIM//5, mask_zero=True))(word_in)
# charcter embedding
char_in = Input(shape=(X_char_train.shape[1],X_char_train.shape[2]), name='char_in')
emb_char=(TimeDistributed(Embedding(input_dim=len(characters),output_dim=10,input_length=max_len_char,mask_zero=True)))(char_in)
char_enc=(TimeDistributed(LSTM(units=EMBED_DIM//5,return_sequences=False)))(emb_char)
# feature embedding
f_in = Input(shape=(x_feature_train.shape[1],x_feature_train.shape[2]), name='f_in')
emb_f=(TimeDistributed(Embedding(input_dim=len(feat),output_dim=10,input_length=features,mask_zero=True)))(f_in)
f_enc=(TimeDistributed(LSTM(units=EMBED_DIM//5,return_sequences=False)))(emb_f)
# concatenating them word+char+features
x = concatenate([emb_word, char_enc, f_enc])
#passing to Bi-LSTM layer
o=(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True, dropout=0.2)))(x)
#Self attention layer
'''o = (SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))(o)
'''
#CRF layer
crf = CRF(len(class_labels), sparse_target=True,name='crf')
o=(crf)(o)
model=Model(input=[word_in,char_in,f_in],output=o)
model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
model.fit([X_train,np.array(X_char_train).reshape(len(X_char_train),max_len,max_len_char),x_feature_train.reshape(len(x_feature_train),max_len,features)],np.array(y_train).reshape(len(y_train), max_len, 1),batch_size=32, epochs=EPOCHS, validation_split=0.1, verbose=1)
#validation
l4=[0]*max_len_char
l4=numpy.asarray(l4)
y= model.predict([X_test,np.array(X_char_test).reshape((len(X_char_test),max_len, max_len_char)),x_pos_test.reshape(len(x_pos_test),max_len),x_feature_test.reshape(len(x_feature_test),max_len,features)])
y= y.argmax(-1)
l4=[0]*max_len_char
l4=numpy.asarray(l4)
y_pred=[]
test_y_true=[]
ctest_y_pred=[]
ctest_y_true=[]
#removing padding from the validation
for i in range(len(X_test)):
    l=[]
    l1=[]
    for j in range(len(X_test[i])):
        if (X_test[i][j]==0):
            continue
        l.append(y[i][j])
        l1.append(y_test[i][j][0])
        ctest_y_pred.append(y[i][j])
        ctest_y_true.append(y_test[i][j][0])
    y_pred.append(l)
    test_y_true.append(l1)
#converting to numpy array
test_y_pred=numpy.asarray(y_pred)
test_y_true=numpy.asarray(test_y_true)
#writing output for validation
f1=open(output_file,'w')
for i in range(len(test_y_pred)):
    for j in range(len(test_y_pred[i])):
        s='|'+'\t'+'NN'+'\t'+str(class_labels[test_y_pred[i][j]])+'\t'+str(class_labels[test_y_true[i][j]])
        f1.write(s+'\n')
    f1.write('\n')
f1.flush()
f1.close()
print('\n---- Result of BiLSTM-CRF ----\n')
classification_report(ctest_y_true, ctest_y_pred, class_labels)