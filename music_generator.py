import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from shutil import copyfile
from music21 import converter, instrument, note, chord, stream, midi
from tqdm import tqdm
import os
import sys
import glob
import math 
import tensorflow as tf

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dropout, LSTM, Activation, Embedding, Dense

# Inspired from: https://github.com/cpmpercussion/creative-prediction/blob/master/notebooks/3-zeldic-musical-RNN.ipynb

DURATION = 0.25
MELODY_NOTE_OFF = 128 # (stop playing all previous notes)
MELODY_NO_EVENT = 129 # (no change from previous event)

def transform_element(element):
  """ Transform music21 Note or Chord element into array form
  """
  if isinstance(element, note.Note):
    return [np.round(element.offset / DURATION),
            np.round(element.quarterLength / DURATION),
            element.pitch.midi]

  return [np.round(element.offset / DURATION),
          np.round(element.quarterLength / DURATION),
          element.sortAscending().pitches[-1].midi]

def is_note_or_chord(element):
  return isinstance(element, (note.Note, chord.Chord))

def parse_to_df(midi):
  stream_info = np.array([transform_element(e) for e in midi.flat if is_note_or_chord(e)],
                         dtype=np.int)
  df = pd.DataFrame({'offset': stream_info[:, 0],
                     'duration': stream_info[:, 1],
                     'pitch': stream_info[:, 2]})
  df = df.sort_values(['offset','pitch'], ascending=[True, False]) # sort the dataframe properly
  df = df.drop_duplicates(subset=['offset']) # drop duplicate value
  return df

def parse_to_np(file):
  song = converter.parse(file)
  df = parse_to_df(song)
  total_length = np.int(np.round(song.flat.highestTime / 0.25))
  # Fill in the output list
  output = np.full(total_length + 1, MELODY_NO_EVENT,  dtype=np.int16)
  for i in range(total_length):
    if not df[df.offset==i].empty:
      n = df[df.offset==i].iloc[0] # pick the highest pitch at each semiquaver
      output[i] = n.pitch # set note on
      if i + n.duration < len(output):
        output[i+n.duration] = MELODY_NOTE_OFF
  return output

def np_to_df(song_data):
  df = pd.DataFrame({'pitch': song_data})
  df['offset'] = df.index
  df = df[df.pitch != MELODY_NO_EVENT].reset_index(drop=True)
  df['duration'] = - df['offset'].diff(-1)
  df = df[:-1]
  df['duration'] = df.duration.astype(np.int16)
  return df[['offset', 'duration', 'pitch']]

def decode_to_stream(song_data, filename=None):
  df = np_to_df(song_data)
  melody_stream = stream.Stream()
  for _, row in df.iterrows():
    if row.pitch == MELODY_NO_EVENT or row.pitch == MELODY_NOTE_OFF:
      new_note = note.Rest()
    else:
      new_note = note.Note(row.pitch)
    new_note.quarterLength = row.duration * 0.25
    melody_stream.append(new_note)
  if filename:
    melody_stream.write('midi', fp=f'./music_data/output/{filename}')
  return melody_stream

# %%
MINIMUM_NOTE = 0
VOCABULARY_SIZE = 130 - MINIMUM_NOTE
SEQUENCE_SIZE = 30
BATCH_SIZE = 64
HIDDEN_UNITS = 356
TRAIN_FILE = './train_data.npz'

# %%
# Build the decoding model
def detransform(data):
  return data + MINIMUM_NOTE

def create_decode_model():
  decoding_model = Sequential()
  decoding_model.add(layers.Embedding(VOCABULARY_SIZE, HIDDEN_UNITS, batch_input_shape=(1, 1)))
  decoding_model.add(LSTM(HIDDEN_UNITS, stateful=True, return_sequences=True))
  decoding_model.add(LSTM(HIDDEN_UNITS, stateful=True))
  decoding_model.add(Dense(HIDDEN_UNITS // 2))
  decoding_model.add(Dense(VOCABULARY_SIZE, activation='softmax'))
  decoding_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
  return decoding_model

# Inspired from: https://github.com/cpmpercussion/creative-prediction/blob/master/notebooks/3-zeldic-musical-RNN.ipynb
def sample(preds, temperature=1.0):
  """ helper function to sample an index from a probability array"""
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)


def sample_model(seed, model_name, length=500, temperature=1.0):
  '''Samples a musicRNN given a seed sequence.'''
  generated = []
  generated.append(seed)
  next_index = seed
  for i in tqdm(range(length)):
    x = np.array([next_index])
    x = np.reshape(x, (1, 1))
    preds = model_name.predict(x, verbose=0)[0]
    next_index = sample(preds, temperature)
    generated.append(next_index)
  return np.array(generated, dtype=np.int16)

def write_song(seed, weights_file, output_file):
  decoding_model = create_decode_model()
  decoding_model.load_weights(weights_file)
  decoding_model.reset_states() # Start with LSTM state blank
  ai_music = detransform(sample_model(seed - MINIMUM_NOTE, decoding_model))
  melody_stream = decode_to_stream(ai_music)
  melody_stream.write('midi', fp=output_file)
  return melody_stream

if len(sys.argv) != 2:
  print("Help: python music_generator.py {trained_epochs}")

num = sys.argv[1]
melody = write_song(
    64,
    f'./checkpoints/final-train-{num}.h5',
    f'./results/song-{num}.mid')
print(f'song-{num}.mid was created in the results folder')
