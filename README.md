# COMP 4107 - Final project

## Name: Tri Cao
## Student number: 100971065

### List of files

- `checkpoints` directory: includes all the checkpoints of training the LTSM network. They can be loaded to generate the soundtracks without training again
- `results` directory: contains the generated music tracks from the models. Can be played by a MIDI music reader (ex: musescore)
- `data` directory: contains some original songs that was used to generate the training dataset (in MIDI files). Also contains `game_piano_train.npz`, which is the processed training dataset stored in numpy format. It can be loaded directly to train with the model
- `data_preprocessing.py`: script to generate training dataset from midi songs
- `training.py`: script contains the LSTM network declaration and training process
- `music_generation.py`: script to generate music using the trained model

### Running instructions

#### data_preprocessing.py

- Run the command `python data_preprocessing.py` to generate an example training dataset from the 5 songs in the `data` folder

### training.py

- Run the command `python training.py {num_epochs}` to train the network for the specified number of epochs.

Example: `python training.py 5` will train the LSTM network for 5 epochs

### music_generation.py

- Run the command `python training.py {trained_epochs}` to use the model checkpoint after training for
specific number of epochs

Example: `python training.py 20` generate a tracks from the 20 epochs checkpoints

Note: we support checkpoints after 1, 2, 3, 5, 10, 20, 40, 60 and 80 epochs
