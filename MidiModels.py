from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import keras.models
from Constants import *
from sklearn.model_selection import train_test_split as tts
import numpy as np
import pickle
import MidiData
import MidiIO
import time
import keras.backend as K

def load_model(model_name, version='latest', sliding_window_inc=None, new_build=False):
    """
    Attempts to load the given model and version.
    Models should be stored in file names of integers with the extension '.h5'
    Extra kwargs passed to the built model
    :param model_name: the name of the model to load
    :param version: what version, either an integer, or 'latest' to do the latest version based on number
    :param sliding_window_inc: overwrite the sliding window inc if desired
    :param new_build: whether or not to rebuild the model from scratch
    :return: the model loaded if possible
    """

    # Load all the files that are possible models as integers
    files = []
    m_dir = os.path.join(W_DIR, model_name)
    for f in os.listdir(m_dir):
        if f[-3:] == '.h5':
            try:
                files.append(int(f[:-3]))
            except ValueError:
                continue
        elif f == 'latest.h5':
            files.append('l')

    # If there are no files
    if len(files) == 0:
        raise ValueError("Could not find any models to load with name: %s  in dir: %s" % (model_name, W_DIR))

    # Find the midi meta data
    if MIDI_META_FILE_NAME not in os.listdir(m_dir):
        raise ValueError("Could not find a midi metadata file: %s  in dir: %s"
                         % (MIDI_META_FILE_NAME, m_dir))
    with open(os.path.join(m_dir, MIDI_META_FILE_NAME), 'rb') as f:
        model_type, ticks, num_secs, swi = pickle.load(f)

    if sliding_window_inc is None:
        sliding_window_inc = swi

    # Load in the old model
    if version == 'latest':
        latest_file = str(sorted(files)[-1]) + '.h5' if 'l' not in files else 'latest.h5'
        loaded = keras.models.load_model(os.path.join(m_dir, latest_file))
    else:
        if str(version) + '.h5' not in os.listdir(m_dir):
            raise ValueError("Could not find model: %s  in dir: %s" % (version, m_dir))
        loaded = keras.models.load_model(os.path.join(m_dir, str(version) + '.h5'))

    # Make the new model
    if model_type == 'rae':
        model = RecurrentAutoEncoder(model_name, (ticks, num_secs, sliding_window_inc))
    else:
        raise ValueError("Unknown model type loaded: %s" % model_type)

    # Finally, transfer the old model's weights to the new model, set the iteration, and return
    if new_build:
        model.build_model()
        model.model.set_weights(loaded.get_weights())
    else:
        model.model = loaded

    model.iteration = sorted(files)[-1]
    return model


class AutoEncoder:

    def __init__(self, curr_model_name, midi_meta_data, checkpoint_midis=True):
        """
        Make all the needed directories.
        :param curr_model_name: the name of this model (the directory to make to store all the info)
        :param midi_meta_data: a tuple of the data that will be trained on:
            - ticks: the number of time steps per seconds (should be an integer)
            - num_secs: the number of seconds per data point
            - sliding_window_inc: if an integer -> use a sliding window over the song, incrementing by
                sliding_window_inc for each new data point.
                if None -> do not use a sliding window, just split the song up into ticks * num_secs chunks and
                    drop the last one if it is too small
        :param checkpoint_midis: if True, will output both training and validation midi files for the network in
            order to checkpoint progress
        """
        self.model_path = os.path.join(W_DIR, curr_model_name)
        self.training_midis_path = os.path.join(W_DIR, TRAINING_MIDIS_PATH)

        self.midi_train_path = os.path.join(W_DIR, curr_model_name, MIDI_TRAIN_DIR_NAME)
        self.midi_val_path = os.path.join(W_DIR, curr_model_name, MIDI_VAL_DIR_NAME)
        self.checkpoint_midis = checkpoint_midis

        for n in [self.model_path, self.midi_train_path, self.midi_val_path, self.training_midis_path]:
            if n is not None and not os.path.exists(n):
                os.makedirs(n)

        self.ticks, self.num_secs, self.sliding_window_inc = midi_meta_data
        self.data = MidiData.load_midi_data(TRAINING_MIDIS_PATH, self.ticks, self.num_secs, self.sliding_window_inc)
        self.encoded = self.decoded = self.model = None
        self.iteration = 0

        self.last_save_time = time.time()
        self.save_secs = 3600

        self.dump_midi()

    def dump_midi(self, name='ae'):
        # Write the midi meta data for future loading
        with open(os.path.join(self.model_path, MIDI_META_FILE_NAME), 'wb') as f:
            pickle.dump((name, self.ticks, self.num_secs, self.sliding_window_inc), f)

    def build_model(self):
        in_act = 'relu'
        last_act = 'sigmoid'
        batch = True
        dp = 0
        opt = 'adam'
        loss = 'binary_crossentropy'

        def lbd(layer, size, b, d, act):
            ret = Dense(size, use_bias=not b, activation=act)(layer)
            if b:
                ret = BatchNormalization()(ret)
            if d is not None and d > 0:
                ret = Dropout(d)(ret)
            return ret

        # The encoder section
        input_img = Input(shape=(int(self.ticks * self.num_secs) * NUM_NOTES,))
        encoded = lbd(input_img, 8192, batch, dp, in_act)
        encoded = lbd(encoded, 4096, batch, dp, in_act)
        encoded = lbd(encoded, 2048, batch, dp, in_act)
        encoded = lbd(encoded, 1024, batch, dp, in_act)
        encoded = lbd(encoded, 512, batch, dp, in_act)
        encoded = lbd(encoded, 256, False, None, in_act)

        decoded = lbd(encoded, 512, batch, dp, in_act)
        decoded = lbd(decoded, 1024, batch, dp, in_act)
        decoded = lbd(decoded, 2048, batch, dp, in_act)
        decoded = lbd(decoded, 4096, batch, dp, in_act)
        decoded = lbd(decoded, 8192, batch, dp, in_act)
        decoded = lbd(decoded, int(self.ticks * self.num_secs) * NUM_NOTES, False, None, last_act)

        self.encoded = encoded
        self.decoded = decoded
        self.model = Model(input_img, decoded)
        self.model.compile(optimizer=opt, loss=loss)

    def train_model(self, num_iters=1000, test_size=0.1, epochs=20, batch_size=512, save_func=None,
                    midi_save_func=None):
        """
        Trains the model for the given number of iterations
        :param num_iters: the number of iterations to train for
        :param test_size: what percent of the dataset to use for validation purposes
        :param epochs: the number of epochs to train for each iteration
        :param batch_size: batch size
        :param save_func: a function that takes an integer input (the iteration) and returns True if this is an
            iteration time step where the network should be saved
            - if None, uses the default save function
        :param midi_save_func: a function that takes an integer input (the iteration) and returns True if this is an
            iteration where the output midis of the network should be saved (assuming checkpoint_midis is True)
            - if None, uses the default midi save function
        """
        if save_func is None:
            save_func = self._save_func
        if midi_save_func is None:
            midi_save_func = self._midi_save_func

        x_train, x_val = tts(np.array(self.data), test_size=test_size, random_state=RANDOM_STATE)
        for i in range(num_iters):
            self.iteration += 1
            self.model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                           validation_data=(x_val, x_val))

            self._save(save_func, x_train[0], x_val[0], midi_save_func)

        # A final save
        self._save(lambda x: True, x_train[0], x_val[0], lambda x: True)

    def _save(self, save_func, x_train_0, x_val_0, midi_save_func):
        """
        Used to save the network if this is a correct iteration
        """

        if save_func(self.iteration):
            self.model.save(os.path.join(self.model_path, "%d.h5" % self.iteration))

        if self.checkpoint_midis and midi_save_func(self.iteration):
            self._save_midis("t_%d.mid" % self.iteration, "v_%d.mid" % self.iteration, x_train_0, x_val_0)

        if time.time() - self.last_save_time > self.save_secs or "latest.h5" not in os.listdir(self.model_path):
            self.model.save(os.path.join(self.model_path, "latest.h5"))
            self._save_midis("t_l.mid", "v_l.mid", x_train_0, x_val_0)

    def _save_midis(self, t_name, v_name, x_train_0, x_val_0):
        p_train = self.model.predict(np.array([x_train_0, ]))[0]
        MidiIO.write_midi(MidiData.decode_notes(p_train, self.ticks),
                          os.path.join(self.midi_train_path, t_name))
        p_val = self.model.predict(np.array([x_val_0, ]))[0]
        MidiIO.write_midi(MidiData.decode_notes(p_val, self.ticks),
                          os.path.join(self.midi_val_path, v_name))
        self._make_gt(x_train_0, x_val_0)

    def _make_gt(self, x_train_0, x_val_0):
        if "tgt.mid" not in os.listdir(self.midi_train_path):
            MidiIO.write_midi(MidiData.decode_notes(x_train_0, self.ticks),
                              os.path.join(self.midi_train_path, "tgt.mid"))
        if "vgt.mid" not in os.listdir(self.midi_val_path):
            MidiIO.write_midi(MidiData.decode_notes(x_val_0, self.ticks),
                              os.path.join(self.midi_val_path, "vgt.mid"))

    @staticmethod
    def _save_func(iteration):
        if iteration < 100 and iteration % 10 == 0:
            return True
        elif iteration < 1000 and iteration % 100 == 0:
            return True
        elif iteration % 300 == 0:
            return True
        return False

    @staticmethod
    def _midi_save_func(iteration):
        if iteration < 50 and iteration % 3 == 0:
            return True
        elif iteration < 500 and iteration % 10 == 0:
            return True
        elif iteration % 50 == 0:
            return True


class RecurrentAutoEncoder(AutoEncoder):

    def dump_midi(self, name='rae'):
        super().dump_midi(name)

    @staticmethod
    def custom_loss(y_pred, y_true):
        return K.mean(K.square(y_pred[:, :, :NUM_NOTES] - y_true[:, :, :NUM_NOTES])) \
            + K.mean(K.square(K.cumsum(y_pred[:, :, NUM_NOTES:], axis=2) - K.cumsum(y_true[:, :, NUM_NOTES:], axis=2)))

    def build_model(self):
        in_act = 'relu'

        # define encoder
        visible = Input(shape=(self.data.shape[1], self.data.shape[2]))
        encoder = LSTM(256, activation=in_act)(visible)
        # define reconstruct decoder
        decoder1 = RepeatVector(self.data.shape[1])(encoder)
        decoder1 = LSTM(256, activation=in_act, return_sequences=True)(decoder1)
        decoder1 = TimeDistributed(Dense(self.data.shape[2]))(decoder1)
        # tie it together
        self.model = Model(inputs=visible, outputs=[decoder1, ])
        self.model.compile(optimizer='adam', loss='cross_entropy')  #RecurrentAutoEncoder.custom_loss

    def train_model(self, num_iters=1000, test_size=0.1, epochs=20, batch_size=512, save_func=None,
                    midi_save_func=None):
        """
        Trains the model for the given number of iterations
        :param num_iters: the number of iterations to train for
        :param test_size: what percent of the dataset to use for validation purposes
        :param epochs: the number of epochs to train for each iteration
        :param batch_size: batch size
        :param save_func: a function that takes an integer input (the iteration) and returns True if this is an
            iteration time step where the network should be saved
            - if None, uses the default save function
        :param midi_save_func: a function that takes an integer input (the iteration) and returns True if this is an
            iteration where the output midis of the network should be saved (assuming checkpoint_midis is True)
            - if None, uses the default midi save function
        """
        if save_func is None:
            save_func = self._save_func
        if midi_save_func is None:
            midi_save_func = self._midi_save_func

        x_train, x_val = tts(np.array(self.data), test_size=test_size, random_state=RANDOM_STATE)
        for i in range(num_iters):
            self.iteration += 1
            self.model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                           validation_data=(x_val, x_val))
            self._save(save_func, x_train[0], x_val[0], midi_save_func)

        # A final save
        self._save(lambda x: True, x_train[0], x_val[0], lambda x: True)
