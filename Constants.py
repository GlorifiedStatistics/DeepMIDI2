import os

W_DIR = "D:/Store/DeepMIDI"  # The directory to do all of the big data stuff in
TRAINING_MIDIS_DIR_NAME = 'mazurka_midis'  # The directory of the MIDI training data
TRAINING_MIDIS_PATH = os.path.join(W_DIR, TRAINING_MIDIS_DIR_NAME)
MIDI_TRAIN_DIR_NAME = 'train_midis'
MIDI_VAL_DIR_NAME = 'val_midis'
MIDI_META_FILE_NAME = 'midi_meta.pkl'

MIDI_NOTE_OFFSET = 20  # Midi note value 20 is A0
NUM_NOTES = 88  # Number of notes to use
RANDOM_STATE = 7342165  # For RNG
