# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:01:00 2022

@author: isaac
"""

import re
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional


class Two():
    def plot_count(self, df, cols):
      for i in cols:
        sns.countplot(x=df[i])
        plt.title(i)
        plt.show()
        
class Three():
    def split_text(self, df_series):
      for index, text in enumerate(df_series):
        df_series[index] = re.sub('([^a-z])([^A-Z])', ' ', text).lower().split()
      return df_series
  
    
class Models():
    def seq_model(self, input_shape, output_shape, vocab_size, 
                  out_dim, layers, nodes, dropout, activation):

      model = Sequential()
      model.add(Input(shape=(input_shape)))
      model.add(Embedding(vocab_size, out_dim))

      for i in range(layers):
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))

      model.add(Bidirectional(LSTM(nodes)))
      model.add(Dropout(dropout))
      model.add(Dense(output_shape, activation=activation))
      return model
        
        
class Evaluation():
    def plot_loss(self, hist, loss, val):
      plt.figure()
      plt.plot(hist.history[loss])
      plt.plot(hist.history[val])
      plt.xlabel('epoch')
      plt.legend(['Training '+loss, 'Validation '+loss])
      plt.show()