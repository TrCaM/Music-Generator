import numpy as np
import sys

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dropout, LSTM, Activation, Embedding, Dense

MINIMUM_NOTE = 0
VOCABULARY_SIZE = 130 - MINIMUM_NOTE
SEQUENCE_SIZE = 30
BATCH_SIZE = 64
HIDDEN_UNITS = 256
TRAIN_FILE = './data/game_piano_train.npz'

def transform_data(data):
  return data - MINIMUM_NOTE

def detransform(data):
  return data + MINIMUM_NOTE

def build_training_sequence(raw_seq):
  if len(raw_seq) < SEQUENCE_SIZE:
    return []
  X = []
  Y = []
  for i in range(SEQUENCE_SIZE, len(raw_seq)):
    X.append(raw_seq[i-SEQUENCE_SIZE:i])
    Y.append(raw_seq[i])
  return np.array(X), np.array(Y)

def load_data(train_file):
  with np.load(train_file, allow_pickle=True) as data:
    train_set = data['train']

  train_X, train_Y = np.array([], dtype=np.int16).reshape(0, 30), np.array([])
  for song in train_set:
    song = transform_data(song)
    if len(song) < SEQUENCE_SIZE:
      continue
    X, Y = build_training_sequence(song)
    train_X = np.concatenate((train_X, X))
    train_Y = np.concatenate((train_Y, Y))

  return train_X, train_Y

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Help: python training.py {num_epochs}")

  print("Loading dataset")
  train_X, train_Y = load_data(TRAIN_FILE)
  print(train_X.shape)
  print(train_Y.shape)

  # Build the 2-layer LSTM model
  print("Building model")
  model = Sequential()
  model.add(layers.Embedding(VOCABULARY_SIZE, HIDDEN_UNITS, input_length=SEQUENCE_SIZE))
  model.add(LSTM(HIDDEN_UNITS, return_sequences=True))
  model.add(Dropout(0.3))
  model.add(LSTM(HIDDEN_UNITS))
  model.add(Dropout(0.3))
  model.add(Dense(HIDDEN_UNITS // 2))
  model.add(Dropout(0.3))
  model.add(Dense(VOCABULARY_SIZE, activation='softmax'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
  model.summary()

  # Train the model and save the weights
  epochs = int(sys.argv[-1])
  print("Start training...")
  model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=epochs)
  print("Save checkpoint.")
  model.save_weights(f'checkpoints/training-checkpoints-{epochs}.h5')
