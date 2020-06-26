import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from shutil import copyfile
from music21 import converter, instrument, note, chord, stream, midi
from tqdm import tqdm
import os
import glob
import math 
import tensorflow as tf

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dropout, LSTM, Activation, Embedding, Dense

# %% [markdown]
# # Analyze music dataset

# %% [markdown]
# ## Utility functions to transform midi file to dataframe format
# We iterate through all element (notes and chords) in the midi file to encode information:
#   - offset: the time offset of the element from the start of the piece (measure as the
#             number semiquavers)
#   - duration: the duration of the element
#   - pitch: the pitch of the note or the highest note in the chord if the element is a chord
#
# Simplify the data for training by keeping only the highest note for each offset

# %%
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

# ## Now load all training songs into numpy format
files = glob.glob("./data/*.mid")
song_data = []
for file in tqdm(files):
  song_data.append(parse_to_np(file))
np_song_data = np.array(song_data)
np.savez('./data/example_train_data.npz', train=np_song_data)

