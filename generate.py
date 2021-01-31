# Standard Data Science Libraries
import pickle
import math
import pandas as pd
import numpy as np
from numpy import array

# Neural Net Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Neural Net Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

# Neural Net Training
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

#Load tokenizer object
with open('tokenizer.pkl', 'rb') as token:
    tokenizer = pickle.load(token)
#Load dictionary
with open('config.dictionary', 'rb') as token:
    reverse_word_map = pickle.load(token)
#Load model
model = load_model('model.hdf5')

#Predict the next text of the sequence
def gen(model,seq,max_len):
    # Tokenize the input string
    tokenized_sent = tokenizer.texts_to_sequences([seq])
    #Total length of text will be max_len
    max_len = max_len+len(tokenized_sent[0])

    while len(tokenized_sent[0]) < max_len:
        #Select last 19 token of the sequence and predict the next shape
        padded_sentence = pad_sequences(tokenized_sent[-19:],maxlen=19)
        #reshape(1, -1) means we have 1 row and unknown number of column
        #op is 1d array gives probablity of occurance of each word argmax() function return index of maximum value in array
        op = model.predict(np.asarray(padded_sentence).reshape(1,-1))
        tokenized_sent[0].append(op.argmax()+1)

    return " ".join(map(lambda x : reverse_word_map[x],tokenized_sent[0]))


#Print generated text
def test_models(test_string,sequence_length= 50, model = model):
    print('Input String: ', test_string)
    print(gen(model,test_string,sequence_length))

#Give input sequence
input_sequence = input("Enter string: ")
test_models(input_sequence)
