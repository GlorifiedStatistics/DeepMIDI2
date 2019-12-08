from Constants import *
import MidiData
import MidiIO
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization

tps = 36
seq_length = 100
checkpoint_midi_length = 1000
model_name = 'rnn_tick_big'
midi_type = 'tickwise'
load = 12

space = -1000

if midi_type == 'tickwise':
    reader = MidiData.TickwiseReader(tps)
elif midi_type == 'update':
    reader = MidiData.MidiUpdateReader(tps, 50)
else:
    raise ValueError("Unknown midi_type: %s" % midi_type)


def one_hot(value, features):
    return [1 if i == value else 0 for i in range(features)]


def from_one_hot(bits):
    return np.where(bits == 1)[0][0]


def from_network(vals):
    ret = np.zeros(vals.shape)
    ret[0, np.argmax(vals[0])] = 1
    return ret


print("Reading in all data...")

# Read in all of the data
total_data = []
for f in [f for f in os.listdir(TRAINING_MIDIS_PATH) if f[-4:] == '.mid']:
    total_data.append(reader.from_messages(MidiIO.read_midi(os.path.join(TRAINING_MIDIS_PATH, f))))

print("Finding max/min...")

# Make the note map
note_map = []
for f in total_data:
    for l in f:
        for v in l:
            if v not in note_map:
                note_map.append(v)
note_map = [space] + list(sorted(note_map))

print("Converting data...")

# Make all of the data
enc_data = []
for f in total_data:
    app = []
    for l in f:
        # Convert to one-hot
        for v in l:
            app.append(one_hot(note_map.index(v), len(note_map)))
        app.append(one_hot(note_map.index(space), len(note_map)))
    enc_data.append(np.array(app))

print("Doing sliding window, making x & y...")

x = []
y = []

for f in enc_data:
    for i in range(len(f) - seq_length):
        x.append(f[i:i + seq_length])
        y.append(f[i + seq_length])

x = np.array(x)
y = np.array(y)

print("Shapes:", x.shape, y.shape)

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1, random_state=RANDOM_STATE)
dp = 0.4


model = Sequential()
model.add(LSTM(512, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(dp))
model.add(LSTM(512, return_sequences=True))
#model.add(Dropout(dp))
model.add(LSTM(512))
#model.add(Dropout(dp))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(dp))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(dp))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='nadam', loss='categorical_crossentropy')

if load is not None:
    model.load_weights(os.path.join(W_DIR, model_name+"/%d_model.h5" % load))

print(model.summary())

if not os.path.exists(os.path.join(W_DIR, model_name)):
    os.mkdir(os.path.join(W_DIR, model_name))
if not os.path.exists(os.path.join(W_DIR, model_name + "/midis")):
    os.mkdir(os.path.join(W_DIR, model_name + "/midis"))

for i in range((load + 1) if load is not None else 0, 200):
    model.fit(x_train, y_train, batch_size=512, epochs=1, validation_data=(x_test, y_test))
    model.save(os.path.join(W_DIR, model_name + "/%d_model.h5" % i))

    # Make the midi for this model currently
    curr_state = x_test[0:1]
    result = []

    for k in range(checkpoint_midi_length):
        output = model.predict(curr_state)
        z_output = from_network(output)
        result.append(z_output[0])
        curr_state[0] = np.append(curr_state[0], z_output, axis=0)[1:]

    note_data = []
    curr_notes = []
    for row in result:
        v = note_map[from_one_hot(row)]
        if v == space:
            note_data.append(curr_notes)
            curr_notes = []
        else:
            if v not in curr_notes:
                curr_notes.append(v)

    MidiIO.write_midi(reader.to_messages(note_data),
                      os.path.join(W_DIR, model_name + "/midis/%d_checkpoint.mid" % i))
