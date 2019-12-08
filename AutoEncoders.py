import MidiIO
from Constants import *
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, RepeatVector, TimeDistributed, \
    SimpleRNN, GRU, Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from sklearn.model_selection import train_test_split as tts
import numpy as np
import time


class BaseModel:

    def __init__(self, curr_model_name, checkpoint_midis=True, save_secs=3600):
        """
        The base model for a network
        :param curr_model_name: the name of this model (the directory to make to store all the info).
        :param checkpoint_midis: if True, will output both training and validation midi files for the network in
            order to checkpoint progress.
        :param save_secs: the number of seconds to wait before absolutely saving, no matter the iteration.
            Set to None if you don't want this.
        """
        # Get/make paths to all the folders
        self.model_path = os.path.join(W_DIR, curr_model_name)
        self.training_midis_path = os.path.join(W_DIR, TRAINING_MIDIS_PATH)
        self.midi_train_path = os.path.join(W_DIR, curr_model_name, MIDI_TRAIN_DIR_NAME)
        self.midi_val_path = os.path.join(W_DIR, curr_model_name, MIDI_VAL_DIR_NAME)
        for n in [self.model_path, self.midi_train_path, self.midi_val_path, self.training_midis_path]:
            if n is not None and not os.path.exists(n):
                os.makedirs(n)

        # Params
        self.checkpoint_midis = checkpoint_midis

        # Presets
        self.model = self.data = self.write_func = None
        self.midi_reader = self.x_train = self.x_test = self.y_train = self.y_test = None
        self.iteration = self.last_save_time = 0
        self.save_secs = save_secs

    def build_model(self, data, **kwargs):
        """
        Builds the model on the given data.
        :param data: the data to train on
        :param kwargs: kwargs for future implementations
        """
        raise NotImplementedError("build_model has not been implemented yet...")

    def training_session(self, midi_reader, iterations=1000, data=None, test_size=0.1, epochs=20, batch_size=512,
                         reader_kwargs=None, build_kwargs=None, load=None, model_name=''):
        """
        Builds this network on the data from the given midi_reader (or on data if it is specified). Then, trains
            the network for the given number of iterations, saving periodically.
        :param midi_reader: the MidiData reader to encode/decode messages that can be written to .mid files
        :param iterations: the number of iterations to train
        :param data: the data to train on. If left as None, calls midi_reader.get_all_midi_data() for data
        :param test_size: percent of dataset to use for validation
        :param epochs: the number of epochs per iteration
        :param batch_size: the batch size
        :param reader_kwargs: the kwargs to pass to the midi_reader when reading in data
        :param build_kwargs: the kwargs to pass to the self.build_model() function
        """
        if data is None:
            data = midi_reader.get_all_midi_data(**reader_kwargs if reader_kwargs is not None else {})
        self.midi_reader = midi_reader
        self.build_model(data, **build_kwargs if build_kwargs is not None else {})
        self.x_train, self.x_test, self.y_train, self.y_test = tts(self.data[0], self.data[1], test_size=test_size,
                                                                   random_state=RANDOM_STATE)
        if load is not None:
            self.model.load_weights(os.path.join(W_DIR, model_name + "/%d.h5" % load))
            self.iteration = load + 1

        print(self.model.summary())

        for i in range(iterations):
            print("\nStarting iteration:", self.iteration, "\n")
            self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                           validation_data=(self.x_test, self.y_test))
            self.save_curr_state()
            self.iteration += 1

    def save_curr_state(self):
        """
        Checks whether or not we should save on this iteration, then saves the model, and if checkpoint_midis is
            True, saves midis (assuming save_midis() has been overridden)
        """
        if self._save_func(self.iteration):
            self.model.save(os.path.join(self.model_path, "%d.h5" % self.iteration))

        if self.checkpoint_midis and self._midi_save_func(self.iteration):
            self.save_midis("t_%d.mid" % self.iteration, "v_%d.mid" % self.iteration)

        if time.time() - self.last_save_time > self.save_secs or "latest.h5" not in os.listdir(self.model_path):
            self.model.save(os.path.join(self.model_path, "latest.h5"))
            if self.checkpoint_midis:
                self.save_midis("t_l.mid", "v_l.mid")

    def get_midis_to_save(self, **kwargs):
        """
        Should return a 4-tuple of (test_ground_truth, test_prediction, validation_ground_truth, validation_prediction)
        :param kwargs: extra kwargs for future implementations
        """
        raise NotImplementedError("save_midis has not been implemented yet...")

    def save_midis(self, t_name, v_name):
        """
        Saves the midis and ground truth examples.
        :param t_name: the name for the training midis
        :param v_name: the name for the validation midis
        """
        tgt, t_pred, vgt, v_pred = self.get_midis_to_save()
        MidiIO.write_midi(
            self.midi_reader.to_messages(self.midi_reader.from_encoded(self.midi_reader.from_network(t_pred))),
            os.path.join(self.midi_train_path, t_name))
        MidiIO.write_midi(
            self.midi_reader.to_messages(self.midi_reader.from_encoded(self.midi_reader.from_network(v_pred))),
            os.path.join(self.midi_val_path, v_name))

        # The ground truth, we do not need "from_network()" here
        if "tgt.mid" not in os.listdir(self.midi_train_path):
            MidiIO.write_midi(
                self.midi_reader.to_messages(self.midi_reader.from_encoded(self.midi_reader.from_network(tgt))),
                os.path.join(self.midi_train_path, "tgt.mid"))
        if "vgt.mid" not in os.listdir(self.midi_val_path):
            MidiIO.write_midi(
                self.midi_reader.to_messages(self.midi_reader.from_encoded(self.midi_reader.from_network(vgt))),
                os.path.join(self.midi_val_path, "vgt.mid"))

    @staticmethod
    def _save_func(iteration):
        if iteration < 100 and iteration % 10 == 0:
            return True
        elif iteration < 1000 and iteration % 10 == 0:
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


class AutoEncoder(BaseModel):

    def __init__(self, curr_model_name, checkpoint_midis=True, save_secs=3600):
        """
        A default MLP AutoEncoder
        """
        super().__init__(curr_model_name, checkpoint_midis, save_secs)
        self.encoder = self.decoder = None

    def build_model(self, data, **kwargs):
        """
        Builds the model on the given data.
        :param data: a 2-d numpy array of the data for this network where each row is a given input
        :param kwargs: kwargs for the build. For this model, can be:
            - in_act: the inside activation function
                *default*: 'relu'
            - last_act: the activation function on the last layer
                *default*: 'sigmoid'
            - batch_norm: whether or not to use batch normalization
                *default*: True
            - dropout: the dropout percentage. A dp of None or 0 means no dropout
                *default*: 0
            - opt: the optimizer
                *default*: 'adam'
            - loss: the loss function
                *default*: 'binary_crossentropy'
        """
        self.data = data, data

        in_act = kwargs['in_act'] if 'in_act' in kwargs.keys() else 'relu'
        last_act = kwargs['last_act'] if 'last_act' in kwargs.keys() else 'sigmoid'
        batch = kwargs['batch'] if 'batch' in kwargs.keys() else True
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        opt = kwargs['opt'] if 'opt' in kwargs.keys() else 'adam'
        loss = kwargs['loss'] if 'loss' in kwargs.keys() else 'binary_crossentropy'

        def lbd(layer, size, b, d, act):
            ret = Dense(size, use_bias=not b, activation=act)(layer)
            if b:
                ret = BatchNormalization()(ret)
            if d is not None and d > 0:
                ret = Dropout(d)(ret)
            return ret

        # The encoder section
        input_img = Input(shape=(self.data[0].shape[1],))
        encoded = lbd(input_img, 4096, batch, None, in_act)
        encoded = lbd(encoded, 2048, batch, None, in_act)
        encoded = lbd(encoded, 1024, batch, None, in_act)
        encoded = lbd(encoded, 512, batch, None, in_act)
        encoded = lbd(encoded, 256, batch, None, in_act)
        encoded = lbd(encoded, 128, False, None, in_act)

        decoded = lbd(encoded, 256, batch, dropout, in_act)
        decoded = lbd(decoded, 512, batch, dropout, in_act)
        decoded = lbd(decoded, 1024, batch, dropout, in_act)
        decoded = lbd(decoded, 2048, batch, dropout, in_act)
        decoded = lbd(decoded, 4096, batch, dropout, in_act)
        decoded = lbd(decoded, self.data[0].shape[1], False, None, last_act)

        self.model = Model(input_img, decoded)
        self.model.compile(optimizer=opt, loss=loss)

    def get_midis_to_save(self, **kwargs):
        """
        Should return a 4-tuple of (test_ground_truth, test_prediction, validation_ground_truth, validation_prediction)
        :param kwargs: extra kwargs for future implementations
        """
        return np.array([self.x_train[0]]), self.model.predict(np.array([self.x_train[0]])),\
               np.array([self.x_test[0]]), self.model.predict(np.array([self.x_test[0]]))


class ConvolutionalAutoEncoder(AutoEncoder):
    def build_model(self, data, **kwargs):
        """
        Build a convolutional nn
        :param data: the data to use. Should have shape:
            (num_examples_in_batch, num_steps_per_datapoint, datapoint_size)
        :param kwargs: kwargs to use when building the encoder
            - in_act: the activation function for inner layers
                *default*: 'relu'
            - last_act: the activation function for the last layer
                *default*: 'sigmoid'
            - opt: the optimizer
                *default*: 'adam'
            - loss: the loss function
                *default*: 'binary_crossentropy'
        """
        self.data = np.reshape(data, list(data.shape) + [1]), np.reshape(data, list(data.shape) + [1])

        in_act = kwargs['in_act'] if 'in_act' in kwargs.keys() else 'relu'
        last_act = kwargs['last_act'] if 'last_act' in kwargs.keys() else 'sigmoid'
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        opt = kwargs['opt'] if 'opt' in kwargs.keys() else 'adam'
        loss = kwargs['loss'] if 'loss' in kwargs.keys() else 'binary_crossentropy'

        # input = num_examples, ticks * tps, num_notes

        # define encoder
        visible = Input(shape=self.data[0].shape[1:4])
        encoder = Conv2D(32, (3, 3), activation=in_act, padding='same')(visible)
        encoder = MaxPooling2D((2, 2))(encoder)
        encoder = Conv2D(64, (3, 3), activation=in_act, padding='same')(encoder)
        encoder = MaxPooling2D((3, 3))(encoder)
        encoder = Conv2D(64, (3, 3), activation=in_act, padding='same')(encoder)
        encoder = MaxPooling2D((2, 2))(encoder)
        encoder = Conv2D(64, (3, 3), activation=in_act, padding='same')(encoder)
        encoder = MaxPooling2D((2, 2))(encoder)
        encoder = Conv2D(64, (3, 3), activation=in_act, padding='same')(encoder)
        encoder = Flatten()(encoder)
        encoder = Dense(256, activation=in_act)(encoder)

        # define reconstruct decoder
        decoder = Dense(11*9*4*2)(encoder)
        decoder = Reshape((9*4, 11*2, 1))(decoder)
        decoder = Conv2DTranspose(64, (3, 3), strides=2, activation=in_act, padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Conv2DTranspose(64, (3, 3), strides=2, activation=in_act, padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Conv2DTranspose(32, (3, 3), activation=in_act, padding='same')(decoder)
        decoder = Conv2D(1, (3, 3), activation=last_act, padding='same')(decoder)

        # tie it together
        self.model = Model(inputs=visible, outputs=decoder)
        self.model.compile(optimizer=opt, loss=loss)

    def get_midis_to_save(self, **kwargs):
        """
        Should return a 4-tuple of (test_ground_truth, test_prediction, validation_ground_truth, validation_prediction)
        :param kwargs: extra kwargs for future implementations
        """
        return self.x_train[0], self.model.predict(self.x_train[0:1])[0], self.x_test[0], self.model.predict(self.x_test[0:1])[0]



class RecurrentAutoEncoder(AutoEncoder):
    def build_model(self, data, **kwargs):
        """
        Build a recurrent auto encoder model
        :param data: the data to use. Should have shape:
            (num_examples_in_batch, num_steps_per_datapoint, datapoint_size)
        :param kwargs: kwargs to use when building the encoder
            - rec_act: the activation function to use inside the recurrent layers
                *default*: 'tanh'
            - units: the number of neurons per recurrent layer
                *default*: 128
            - layer_type: the type of layer to use:
                'none'/'normal'/None -> plain RNN
                'lstm' -> LSTM network
                'gru' -> GRU network
                *default*: 'lstm'
            - dropout: the dropout to use between layers. If 0 or None, no dropout is used
                *default*: 0
            - rec_dropout: the dropout to use inside recurrent layers. If 0 or None, no dropout is used
                *default*: 0
            - opt: the optimizer
                *default*: 'adam'
            - loss: the loss function
                *default*: 'binary_crossentropy'
        :return:
        """
        self.data = data, data

        rec_act = kwargs['rec_act'] if 'rec_act' in kwargs.keys() else 'tanh'
        units = kwargs['units'] if 'units' in kwargs.keys() else 128
        layer_type = kwargs['layer_type'] if 'layer_type' in kwargs.keys() else 'lstm'
        rec_dropout = kwargs['rec_dropout'] if 'rec_dropout' in kwargs.keys() else 0
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        opt = kwargs['opt'] if 'opt' in kwargs.keys() else 'adam'
        loss = kwargs['loss'] if 'loss' in kwargs.keys() else 'binary_crossentropy'

        def build_layer(prev_layer, rs=False, hs=units, lt=layer_type, rdp=rec_dropout):
            if lt is None or lt.lower() in ['none', 'normal']:
                return SimpleRNN(hs, recurrent_activation=rec_act, return_sequences=rs,
                                 recurrent_dropout=rdp)(prev_layer)
            elif lt.lower() == 'lstm':
                return LSTM(hs, recurrent_activation=rec_act, return_sequences=rs,
                            recurrent_dropout=rdp)(prev_layer)
            elif lt.lower() == 'gru':
                return GRU(hs, recurrent_activation=rec_act, return_sequences=rs,
                           recurrent_dropout=rdp)(prev_layer)
            else:
                raise ValueError("Unknown recurrent layer type:", layer_type)

        # define encoder
        visible = Input(shape=(self.data[0].shape[1], self.data[0].shape[2]))
        encoder = build_layer(visible)
        # define reconstruct decoder
        decoder = RepeatVector(self.data[0].shape[1])(encoder)
        decoder = build_layer(decoder, rs=True)
        #decoder = BatchNormalization()(decoder)
        decoder = TimeDistributed(Dense(self.data[0].shape[2]))(decoder)
        # tie it together
        self.model = Model(inputs=visible, outputs=[decoder, ])
        self.model.compile(optimizer=opt, loss=loss)

    def get_midis_to_save(self, **kwargs):
        """
        Should return a 4-tuple of (test_ground_truth, test_prediction, validation_ground_truth, validation_prediction)
        :param kwargs: extra kwargs for future implementations
        """
        return self.x_train[0], self.model.predict(self.x_train[0:1])[0], self.x_test[0], self.model.predict(self.x_test[0:1])[0]
