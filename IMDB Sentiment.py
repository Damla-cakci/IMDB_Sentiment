import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences  #pad_sequences sayesinde kelime sayısını bir sayıya sabitleriz. # Fixed: Changed import statement
from keras.models import Sequential
from keras.layers.embeddings import Embedding  #bu embedding layerı int leri yoğunluk vektörüne çevirir
from keras.layers import SimpleRNN, Dense, Activation

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path="imdb.npz",  # nonpie zip
                                                      num_words=None,  # veri setinin içindeki kelimeler
                                                      skip_top=0,  # en sık kullanılan kelimeri ignore etmek için
                                                      maxlen=None,  # kelime sayısını belirlemek için
                                                      seed=113,  # karıştırmada aynı ortak sırayı vermesi için
                                                      start_char=1,  # hangi karakterden başlıyacağı
                                                      ov_char=2,  # default değeri 2
                                                      index_from=3)  # defult değeri 3

review_len_train = []
review_len_test = []

for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))

sns.displot(review_len_train, hist_kws={"alpha": 0.3})
sns.displot(review_len_test, hist_kws={"alpha": 0.3})

# number of words
word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

for keys, values in word_index.items():
    if values == 1:  # kelimenin numarası
        print(keys)


def WhatItSay(index=24):
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])
    print(decode_review)
    print(Y_train[index])
    return decode_review


decoded_review = WhatItSay(36)

#veri setini sınırlamak için:
num_words = 15000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)
maxlen = 130
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_lenght = len(X_train[0])))
rnn.add(SimpleRNN(16,input_shape = (num_words,maxlen), return_sequences = False, activation = "relu"))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss = "binary_crossentropy", optimizer="rmsprop",metrics= ["accuracy"])

