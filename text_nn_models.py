#%%
import pandas as pd
import os
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import layers, models, optimizers
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics

#%%
# Data directory
dataDir = os.getcwd() + '/data/'
if not os.path.exists(dataDir):
  os.makedirs(dataDir)

# x: input y:output
# Balanced Training data
FILEPATH = dataDir + 'text_balanced.csv'
df_balanced_train = pd.read_csv(FILEPATH)
df_balanced_train = df_balanced_train.drop(['POST ID'], axis=1)
train_balanced_x = df_balanced_train['post']
train_balanced_y = df_balanced_train['NSFW']

train_balanced_x, valid_balanced_x, train_balanced_y, valid_balanced_y = train_test_split(
  train_balanced_x.values,
  train_balanced_y.values,
  test_size=0.25,
  random_state=1000)


# Representative Training data
FILEPATH = dataDir + 'text_representative.csv'
df_representative_train = pd.read_csv(FILEPATH)
df_representative_train = df_representative_train.drop(['POST ID'], axis=1)
train_representative_x = df_representative_train['post']
train_representative_y = df_representative_train['NSFW']

train_representative_x, valid_representative_x, train_representative_y, valid_representative_y = train_test_split(
  train_representative_x.values,
  train_representative_y.values,
  test_size=0.25,
  random_state=1000)

# Testing data (Always on representative/real life data)
FILEPATH = dataDir + 'text_test.csv'
df_test = pd.read_csv(FILEPATH)
df_test = df_test.drop(['POST ID'], axis=1)
test_x = df_test['post']
test_y = df_test['NSFW']


#%%
# Result directory
resultDir = os.getcwd() + '/results/text/'
if not os.path.exists(resultDir):
  os.makedirs(resultDir)

# Result (Training data size: 10 million and Test data size: 0.1 million)
FILEPATH = resultDir + 'nn_models_result_1M_0.1M.csv'
result_file =  open(FILEPATH, 'w', newline='', encoding='utf-8')
nn_models_result = csv.writer(result_file, delimiter=',')
headers = ["model name", "f1 score", "precision", "recall", "accuracy"]
nn_models_result.writerow(headers)

# Don't forget at later
# result_file.close()




#%%
# label encode the target variable 
encoder = preprocessing.LabelEncoder()

train_balanced_y = encoder.fit_transform(train_balanced_y)
valid_balanced_y = encoder.fit_transform(valid_balanced_y)

train_representative_y = encoder.fit_transform(train_representative_y)
valid_representative_y = encoder.fit_transform(valid_representative_y)

test_y = encoder.fit_transform(test_y)


#%%
tokenizer_balanced = Tokenizer()
tokenizer_balanced.fit_on_texts(train_balanced_x)

tokenizer_representative = Tokenizer()
tokenizer_representative.fit_on_texts(train_representative_x)

# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open(os.getcwd() + '/word_embeddings/glove.6B/glove.6B.100d.txt')):
  values = line.split()
  embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

embeddings_size = 100

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_balanced_seq_x = sequence.pad_sequences(tokenizer_balanced.texts_to_sequences(train_balanced_x), padding='post', maxlen=500)
valid_balanced_seq_x = sequence.pad_sequences(tokenizer_balanced.texts_to_sequences(valid_balanced_x), padding='post', maxlen=500)

train_representative_seq_x = sequence.pad_sequences(tokenizer_representative.texts_to_sequences(train_representative_x), padding='post', maxlen=500)
valid_representative_seq_x = sequence.pad_sequences(tokenizer_representative.texts_to_sequences(valid_representative_x), padding='post', maxlen=500)

test_balanced_seq_x = sequence.pad_sequences(tokenizer_balanced.texts_to_sequences(test_x), padding='post', maxlen=500)
test_representative_seq_x = sequence.pad_sequences(tokenizer_representative.texts_to_sequences(test_x), padding='post', maxlen=500)

# create token-embedding mapping
embedding_matrix_balanced = np.zeros((len(tokenizer_balanced.word_index) + 1, 100))
for word, i in tokenizer_balanced.word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix_balanced[i] = embedding_vector

embedding_matrix_representative = np.zeros((len(tokenizer_representative.word_index) + 1, 100))
for word, i in tokenizer_representative.word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix_representative[i] = embedding_vector



#%%
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

EPOCHS = 2
BATCH_SIZE = 10

def f1_loss(y_true, y_pred):    
  tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
  tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
  fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
  fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

  p = tp / (tp + fp + K.epsilon())
  r = tp / (tp + fn + K.epsilon())

  f1 = 2*p*r / (p+r+K.epsilon())
  f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
  return 1 - K.mean(f1)

def f1_metric(y_true, y_pred):    
  tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
  tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
  fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
  fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

  p = tp / (tp + fp + K.epsilon())
  r = tp / (tp + fn + K.epsilon())

  f1 = 2*p*r / (p+r+K.epsilon())
  f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
  return K.mean(f1)

def train_nn_model(parametrs):
  # fit the training dataset on the classifier
  history = classifier.fit(
    parametrs['train_x'],
    parametrs['train_y'],
    epochs=parametrs['epochs'],
    verbose=parametrs['verbose'],
    validation_data=parametrs['validation_data'],
    batch_size=parametrs['batch_size']
    )
  
  # predict the labels on validation dataset
  predictions = classifier.predict(parametrs['feature_vector_valid'])
  predictions = predictions.argmax(axis=-1)
  
  result = [
    parametrs['model_name'],
    round(metrics.f1_score(predictions, test_y), 3),
    round(metrics.precision_score(predictions, test_y), 3),
    round(metrics.recall_score(predictions, test_y), 3),
    round(metrics.accuracy_score(predictions, test_y), 3)]

  return predictions, history, result

def plot_history(history):
  f1_metric = history.history['f1_metric']
  val_f1_metric = history.history['val_f1_metric']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  x = range(1, len(f1_metric) + 1)

  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(x, f1_metric, 'b', label='Training F1 score')
  plt.plot(x, val_f1_metric, 'r', label='Validation F1 score')
  plt.title('Training and validation F1 score')
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(x, loss, 'b', label='Training loss')
  plt.plot(x, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.savefig(resultDir + '/' + param['model_name'] + '.png')
  plt.close()

#%%
def create_rnn_lstm(word_index, embedding_matrix):
  # Add an Input Layer
  input_layer = layers.Input((500, ))

  # Add the word embedding Layer
  embedding_layer = layers.Embedding(len(word_index) + 1, embeddings_size, weights=[embedding_matrix], trainable=True)(input_layer)
  embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

  # Add the LSTM Layer
  lstm_layer = layers.LSTM(100)(embedding_layer)

  # Add the output Layers
  output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
  output_layer1 = layers.Dropout(0.25)(output_layer1)
  output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

  # Compile the model
  model = models.Model(inputs=input_layer, outputs=output_layer2)
  model.compile(optimizer=optimizers.Adam(), loss=f1_loss, metrics=[f1_metric])#'binary_crossentropy')
  
  return model

def create_gru_lstm(word_index, embedding_matrix):
  # Add an Input Layer
  input_layer = layers.Input((500, ))

  # Add the word embedding Layer
  embedding_layer = layers.Embedding(len(word_index) + 1, embeddings_size, weights=[embedding_matrix], trainable=False)(input_layer)
  embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

  # Add the GRU Layer
  lstm_layer = layers.GRU(100)(embedding_layer)

  # Add the output Layers
  output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
  output_layer1 = layers.Dropout(0.25)(output_layer1)
  output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

  # Compile the model
  model = models.Model(inputs=input_layer, outputs=output_layer2)
  model.compile(optimizer=optimizers.Adam(), loss=f1_loss, metrics=[f1_metric])
  
  return model

def create_bidirectional_rnn(word_index, embedding_matrix):
  # Add an Input Layer
  input_layer = layers.Input((500, ))

  # Add the word embedding Layer
  embedding_layer = layers.Embedding(len(word_index) + 1, embeddings_size, weights=[embedding_matrix], trainable=False)(input_layer)
  embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

  # Add the LSTM Layer
  lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

  # Add the output Layers
  output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
  output_layer1 = layers.Dropout(0.25)(output_layer1)
  output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

  # Compile the model
  model = models.Model(inputs=input_layer, outputs=output_layer2)
  model.compile(optimizer=optimizers.Adam(), loss=f1_loss, metrics=[f1_metric])
  
  return model

def create_rnn(word_index, embedding_matrix):
  # Add an Input Layer
  input_layer = layers.Input((500, ))

  # Add the word embedding Layer
  embedding_layer = layers.Embedding(len(word_index) + 1, embeddings_size, weights=[embedding_matrix], trainable=False)(input_layer)
  embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
  
  # Add the recurrent layer
  rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)
  
  # Add the convolutional Layer
  conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

  # Add the pooling Layer
  pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

  # Add the output Layers
  output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
  output_layer1 = layers.Dropout(0.25)(output_layer1)
  output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

  # Compile the model
  model = models.Model(inputs=input_layer, outputs=output_layer2)
  model.compile(optimizer=optimizers.Adam(), loss=f1_loss, metrics=[f1_metric])
  
  return model

def create_cnn(word_index, embedding_matrix):
  # Add an Input Layer
  input_layer = layers.Input((500, ))

  # Add the word embedding Layer
  embedding_layer = layers.Embedding(len(word_index) + 1, embeddings_size, weights=[embedding_matrix], trainable=False)(input_layer)
  embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

  # Add the convolutional Layer
  conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

  # Add the pooling Layer
  pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

  # Add the output Layers
  output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
  output_layer1 = layers.Dropout(0.25)(output_layer1)
  output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

  # Compile the model
  model = models.Model(inputs=input_layer, outputs=output_layer2)
  model.compile(optimizer=optimizers.Adam(),  loss=f1_loss, metrics=[f1_metric])
  
  return model


#%%
############################ RNN LSTM ##############################

# Balanced
classifier = create_rnn_lstm(
  word_index=tokenizer_balanced.word_index,
  embedding_matrix=embedding_matrix_balanced)

param = {
  'train_x': train_balanced_seq_x,
  'train_y': train_balanced_y,
  'validation_data': (valid_balanced_seq_x, valid_balanced_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "rnn_lstm_balanced"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)

# Representative
classifier = create_rnn_lstm(
  word_index=tokenizer_representative.word_index,
  embedding_matrix=embedding_matrix_representative)

param = {
  'train_x': train_representative_seq_x,
  'train_y': train_representative_y,
  'validation_data': (valid_representative_seq_x, valid_representative_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "rnn_lstm_representative"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)


#%%
############################ GRU LSTM ##############################

# Balanced
classifier = create_gru_lstm(
  word_index=tokenizer_balanced.word_index,
  embedding_matrix=embedding_matrix_balanced)

param = {
  'train_x': train_balanced_seq_x,
  'train_y': train_balanced_y,
  'validation_data': (valid_balanced_seq_x, valid_balanced_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "gru_lstm_balanced"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)

# Representative
classifier = create_gru_lstm(
  word_index=tokenizer_representative.word_index,
  embedding_matrix=embedding_matrix_representative)

param = {
  'train_x': train_representative_seq_x,
  'train_y': train_representative_y,
  'validation_data': (valid_representative_seq_x, valid_representative_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "gru_lstm_representative"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)


#%%
############################ Bidirectional RNN ##############################

# Balanced
classifier = create_bidirectional_rnn(
  word_index=tokenizer_balanced.word_index,
  embedding_matrix=embedding_matrix_balanced)

param = {
  'train_x': train_balanced_seq_x,
  'train_y': train_balanced_y,
  'validation_data': (valid_balanced_seq_x, valid_balanced_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "bidirectional_rnn_balanced"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)

# Representative
classifier = create_bidirectional_rnn(
  word_index=tokenizer_representative.word_index,
  embedding_matrix=embedding_matrix_representative)

param = {
  'train_x': train_representative_seq_x,
  'train_y': train_representative_y,
  'validation_data': (valid_representative_seq_x, valid_representative_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "bidirectional_rnn_representative"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)


#%%
############################ RNN ##############################

# Balanced
classifier = create_rnn(
  word_index=tokenizer_balanced.word_index,
  embedding_matrix=embedding_matrix_balanced)

param = {
  'train_x': train_balanced_seq_x,
  'train_y': train_balanced_y,
  'validation_data': (valid_balanced_seq_x, valid_balanced_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "rnn_balanced"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)

# Representative
classifier = create_rnn(
  word_index=tokenizer_representative.word_index,
  embedding_matrix=embedding_matrix_representative)

param = {
  'train_x': train_representative_seq_x,
  'train_y': train_representative_y,
  'validation_data': (valid_representative_seq_x, valid_representative_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "rnn_representative"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)


#%%
############################ CNN ##############################

# Balanced
classifier = create_cnn(
  word_index=tokenizer_balanced.word_index,
  embedding_matrix=embedding_matrix_balanced)

param = {
  'train_x': train_balanced_seq_x,
  'train_y': train_balanced_y,
  'validation_data': (valid_balanced_seq_x, valid_balanced_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "cnn_balanced"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)

# Representative
classifier = create_cnn(
  word_index=tokenizer_representative.word_index,
  embedding_matrix=embedding_matrix_representative)

param = {
  'train_x': train_representative_seq_x,
  'train_y': train_representative_y,
  'validation_data': (valid_representative_seq_x, valid_representative_y),
  'epochs': EPOCHS,
  'batch_size': BATCH_SIZE,
  'verbose': True,
  'feature_vector_valid': test_balanced_seq_x,
  'test_y': test_y,
  'model_name': "cnn_representative"
  }

predictions, history, result = train_nn_model(param)
plot_history(history)
nn_models_result.writerow(result)



# %%
result_file.close()

# %%
print(metrics.confusion_matrix(test_y, predictions))

