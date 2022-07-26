# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15cabDsog6JFEJR6hcpGfRt8PbCZO-iAo

imports
"""
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from main_module import Two, Evaluation, Three, Models
"""Paths"""

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

OHE_PATH = os.path.join(os.getcwd(), 'models', 'OHE.pkl')
TOKENIZER_PATH = os.path.join(os.getcwd(), 'models', 'TOKENIZER.json')
BEST_MODEL_PATH = os.path.join(os.getcwd(), 'models', 'best_model.h5')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))

"""Step 1 - Data Loading"""

df = pd.read_csv(URL)

df.head()

"""Step 2 - Data Inspection & Visualization
- Target = category
"""

df.info()

df.describe()

df.isna().sum()

step = Two()
step.plot_count(df, cols=['category'])

"""Category is quite balanced
- maybe cut off around tech value will improve the models

Step 3 - Data Cleaning
"""

df2 = df.copy()

df2.duplicated().sum()

"""This section is to discussed on how dropping duplicated makes this project 
   unable to performed well
- After dropping duplicated, it can be observed that the last row of dataframe 
  were not split into respective words instead of grouped as whole
- This is the main reason to why we dont drop duplicated to ensure that all 
  word in dataframe is split individually.
"""

df_temp = df2.drop_duplicates()

df_temp.duplicated().sum()

# splitting text
step = Three()
df_temp = step.split_text(df_temp.text)

# the first 5 rows of dataframe is split
df_temp.head(5)

# re method fail to split on the lower part of dataframe
df_temp.tail(5)

# based on manual observation, it is seen that the regex method wont work on 
# index 2126 onwards
df_temp 

texts = df.text[2126:] # assigning the un-regex to new sets of variable
texts

# splitting text
texts = step.split_text(texts)

# the variables still persist to fail, there this conclude that dropping may 
# bring more harm to the input compared to not drop since the duplicates 
# percentage too is too low to really affect the models
texts[:10] 

"""Splitting each text without dropping duplicates"""

df2.head(10) # inspect head dataframe

df2.tail(10) # inspect tail dataframe

len(df2.text) # inspect text columns

# splitting text
df2.text = step.split_text(df2.text)

df2.head(10) # inspect head dataframe

df2.tail(10) # inspect tail dataframe

"""Although there are funny character like % and currency symbol, most of them 
divided into politics and sport, maybe this will improve our models accuracy

Step 4 - Feature Selection
- There is no need for feature selection due to only 1 features provided

Step 5 - Data Preprocessing
"""
# splitting X and y
category = df2.category
text = df2.text

# vocab size
vocab_size = 20000 # 10000
oov_token = '<OOV>'

# tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

# fit to text
tokenizer.fit_on_texts(df2.text)
word_index = tokenizer.word_index

list(word_index.items())[:50]

# To convert into number
text_int = tokenizer.texts_to_sequences(df2.text)

# getting the mean for count word in every text
mean = []
for i in range(len(text_int)):
    mean.append(len(text_int[i]))

# either to choose mean or median as max_len
max_len = np.median(mean)
# max_len = np.mean(mean)

print('The mean value for word count is :', np.mean(mean))
print('The median value for word count is :', np.median(mean))

# post padding
padded_text = pad_sequences(text_int,
                       maxlen=int(max_len),
                       padding='post',
                       truncating='post')


"""One Hot Encoder for target"""
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category, axis=-1))

# split train and test datasets
X_train, X_test, y_train, y_test = train_test_split(padded_text, category,
                                                    test_size=0.3,
                                                    random_state=42)


"""Model Development"""
# deep learning additional info
input_shape = np.shape(X_train)[1:]
output_shape = df.category.nunique()
out_dim = 256
layers = 2
nodes = 128
dropout = 0.3
activation = 'softmax'

# create deep learning model - LSTM
step = Models()
model_1 = step.seq_model(input_shape, output_shape, vocab_size,
                         out_dim, layers, nodes, dropout, activation)

# model summary
model_1.summary()

# model complime
model_1.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['acc'])

# model architecture
plot_model(model_1,
           show_layer_names=True,
           show_shapes=True)

# callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)


# checkpoint and saving deep_learning model
mdc = ModelCheckpoint(BEST_MODEL_PATH,
                      monitor='val_acc',
                      save_best_only=True,
                      mode='max',
                      verbose=1)

# model training
hist = model_1.fit(X_train, y_train,
                   epochs=30, # training iteration
                   validation_data=(X_test, y_test),
                   callbacks=[tensorboard_callback, mdc])

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard
# %tensorboard --logdir logs


"""Training and Validation loss plots"""

print(hist.history.keys())

step = Evaluation()
step.plot_loss(hist, 'loss', 'val_loss')
step.plot_loss(hist, 'acc', 'val_acc')


"""Classification Report"""

labels = list(df.category.unique())
y_pred = np.argmax(model_1.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
cr = classification_report(y_true, y_pred, target_names = labels)
print(cr)


"""Confusion Matrix"""

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm ,display_labels =labels)
disp.plot(cmap='BuPu')
plt.show()


"""Tokenizing and pickle"""

# json
token_json = tokenizer.to_json()
with open(TOKENIZER_PATH, 'w') as file:
  json.dump(token_json, file)

# pickles
with open(OHE_PATH, 'wb') as file:
  pickle.dump(ohe, file)