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

import pickle


# Import the data
train_df = pd.read_csv('train.csv')
author = train_df[train_df['author'] == 'EAP']["text"]


max_words = 50000 # Max size of the dictionary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(author.values)
sequences = tokenizer.texts_to_sequences(author.values)


# Flatten the list of lists resulting from the tokenization. This will reduce the list to one dimension, allowing us to apply the sliding window technique to predict the next word
text = [item for sublist in sequences for item in sublist]
vocab_size = len(tokenizer.word_index)


sentence_len = 20
pred_len = 1
train_len = sentence_len - pred_len
seq = []

# Sliding window to generate train data
for i in range(len(text)-sentence_len):
    seq.append(text[i:i+sentence_len])

# Reverse dictionary to decode tokenized sequences back to words
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

#Save dictionary and tokenizer object
with open('config.dictionary', 'wb') as config_dictionary_file:
  pickle.dump(reverse_word_map, config_dictionary_file)
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

trainX = []
trainy = []
for i in seq:
    trainX.append(i[:train_len])
    trainy.append(i[-1])

# define model
model_3 = Sequential([
    Embedding(vocab_size+1, 50, input_length=train_len),
    LSTM(150, return_sequences=True),
    LSTM(150),
    Dense(150, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.fit(np.asarray(trainX),
         pd.get_dummies(np.asarray(trainy)),
         epochs = 500,
         batch_size = 10240,
         callbacks = callbacks_list,
         verbose = 2)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(np.asarray(trainX), pd.get_dummies(np.asarray(trainy)), batch_size=128, epochs=100)

model.save('model.hdf5')
