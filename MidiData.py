from Constants import *
from mido import Message
import MidiIO
import pickle
import numpy as np
from PythonUtils import *
import hashlib


def _rolling_window(arr, window_size, window_inc, padding=None):
    """
    Does a rolling window over arr along the 0th axis
    :param arr: the array
    :param window_size: the size of the window
    :param window_inc: the amount to increase the location of the window each step
    :param padding: if padding is not None: pad with this value before doing rolling window
        (pads window_size - 1)
    :return: a numpy array
    """
    arr = np.array(arr)

    if padding is not None:
        pad = np.array(np.full([window_size - 1] + list(arr.shape[1:]), padding))
        arr = np.append(np.append(pad, arr, axis=0), pad, axis=0)

    ret = []
    i = 0
    while i + window_size <= len(arr):
        ret.append(arr[i:i + window_size])
        i += window_inc
    a = np.array(ret)
    return a


def _time_as_total_ticks(messages, ticks_per_sec):
    """
    Changes the 'time' attribute of every message in messages to be the number of ticks from the beginning
        of the song
    :param messages: the list of mido messages
    :param ticks_per_sec: the number of ticks per second
    """
    curr_time = 0
    for m in messages:
        curr_time += m.time
        m.time = round(curr_time * ticks_per_sec)


def _time_as_incremental_ticks(messages, ticks_per_sec):
    """
    Changes the 'time' attribute of every message in messages to be the number of ticks since the last update.
    Calls _time_as_total_ticks in order to make the values more accurate (we don't lose as much information
        when converting)
    :param messages: the list of mido messages
    :param ticks_per_sec: the number of ticks per second
    """
    _time_as_total_ticks(messages, ticks_per_sec)
    curr_time = 0
    for m in messages:
        t = m.time
        m.time -= curr_time
        curr_time = t


####################################
# Bit fillers for MidiUpdateReader #
####################################


def _wait_bit_fill_enc(time, wait_bits):
    """
    Fill 1's from left to right until time bits are filled, the rest are 0's
    """
    if time > wait_bits:
        return [1] * wait_bits, time - wait_bits
    return [1] * time + [0] * (wait_bits - time), 0


def _wait_bit_fill_dec(bits):
    """
    Decode a fill of 1's from left to right, the rest should be 0's
    """
    return sum(bits)


def _wait_bit_single_enc(time, wait_bits):
    """
    Insert single 1 at the index of time to wait, the rest are 0's
    """
    if time > wait_bits:
        return [0] * (wait_bits - 1) + [1], time - wait_bits
    ret = [0] * wait_bits
    if time == 0:
        return ret, 0
    ret[time - 1] = 1
    return ret, 0


def _wait_bit_single_dec(bits):
    """
    Decode a single bit at the index of the time
    """
    try:
        return bits.index(1) + 1
    except ValueError:
        return 0


def _wait_bit_binary_enc(time, wait_bits):
    """
    Make it a binary number using wait_bits bits
    """
    if time > 2 ** wait_bits - 1:
        return [1] * wait_bits, time - 2 ** wait_bits + 1
    ret = [0] * wait_bits
    for i in reversed(list(range(wait_bits))):
        if 2 ** i <= time:
            ret[-(i + 1)] = 1
            time -= 2 ** i
    return ret, 0


def _wait_bit_binary_dec(bits):
    """
    Decode from binary reading right to left
    """
    ret = 0
    for i in range(len(bits)):
        if bits[-(i + 1)] == 1:
            ret += 2 ** i
    return ret


###############
# MidiReaders #
###############


class MidiReader:
    def __init__(self, ticks_per_sec, num_notes=NUM_NOTES, midi_note_offset=MIDI_NOTE_OFFSET):
        """
        An unimplemented data reader
        :param ticks_per_sec: the number of updates per second
        :param num_notes: the number of possible notes that could be hit
        :param midi_note_offset: the value on which A0 starts in midi
        """
        if ticks_per_sec < 1 or int(ticks_per_sec) - ticks_per_sec != 0:
            raise ValueError("ticks must be an integer >= 1")
        if num_notes < 1 or int(num_notes) - num_notes != 0:
            raise ValueError("num_notes must be an integer >= 1")

        self.ticks = ticks_per_sec
        self.num_notes = num_notes
        self.midi_offset = midi_note_offset
        self.reader_type = 'unimplemented'

        self.swi = self.secs_per_dp = self.padding = None

    def get_params(self):
        """
        Returns a tuple of all of the parameters that describe this MidiReader
        """
        return {"reader_type": self.reader_type, "ticks_per_second": self.ticks, "num_notes": self.num_notes,
                "midi_offset": self.midi_offset}

    def get_all_midi_data(self, path=TRAINING_MIDIS_PATH, load=False, save=True, **kwargs):
        """
        Reads in all of the midi data from each midi file in the given path
        :param path: the folder that contains all of the .mid files
        :param load: if True, will attempt to load the midi data from a precomputed file
        :param save: if True, will save this data so it need not be computed in the future. The data is saved
            pre-augmentation as the augmentation step can be done after loading
        :param kwargs: extra kwargs to be used in this function, or passed to future functions
        :return: a numpy array of all the data
        """
        # Load from file if we can
        params = add_dicts(self.get_params(), kwargs)
        file_name = hashlib.sha3_256(str.encode(str(params))).hexdigest()[:10] + ".pkl"
        if load:
            print("Attempting to load data with params: ")
            print(params)

            if file_name in os.listdir(path):
                print("Found file %s, loading in data" % file_name)
                with open(os.path.join(path, file_name), 'rb') as f:
                    return self.augment_data(pickle.load(f), **kwargs)

            print("Could not find file, making new")

        print("Reading in MIDI files...")
        good_files = [f for f in os.listdir(path) if f[-4:] == '.mid']
        total_data = [self.to_encoded(self.from_messages(MidiIO.read_midi(os.path.join(path, f)))) for f in good_files]

        # Save the data if save
        if save:
            print("Saving data with params: ")
            print(params)
            print("To file:", file_name)

            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(total_data, f)

        # Return data
        return self.augment_data(total_data, **kwargs)

    def augment_data(self, data, **kwargs):
        """
        Moves a rolling window over the data to augment and get more
        :param data: the data to augment, must be a list of multiple encoded datas
        :param kwargs: extra kwargs for this and future implementations of augment_data. Here, can have values:
            - secs_per_dp: the number of seconds per data point. If a float, the number of ticks used per
                data point will be round(num_secs * ticks)
            - swi: the sliding_window_increment. Used to augment dataset to get more data via a sliding
                window over the midi with this as its increment
            - padding: if not None - pad the beginning and end with this value before doing sliding window
        """
        self.secs_per_dp = kwargs['secs_per_dp'] if 'secs_per_dp' in kwargs.keys() else 4
        tpdp = round(self.ticks * self.secs_per_dp)  # Ticks per data point

        self.swi = kwargs['swi'] if 'swi' in kwargs.keys() else tpdp
        self.padding = kwargs['padding'] if 'padding' in kwargs.keys() else None

        if self.swi is not None and (self.swi < 1 or int(self.swi) - self.swi != 0):
            raise ValueError("swi must be an integer >= 1")
        if self.secs_per_dp <= 0:
            raise ValueError("secs_per_dp must be > 0")

        ret = _rolling_window(data[0], tpdp, self.swi, self.padding)
        for d in data[1:]:
            ret = np.append(ret, _rolling_window(d, tpdp, self.swi, self.padding), axis=0)
        return ret

    def from_messages(self, messages, *args, **kwargs):
        """
        Convert the data from a list of mido messages into an intermediary representation based on this implementation
        :param messages: the list of mido messages to encode
        :param args: args that could be passed to later implementations
        :param kwargs: kwargs that could be passed to later implementations
        :return: the messages in an intermediary form
        """
        raise NotImplementedError("from_messages has not been implemented yet...")

    def to_messages(self, note_data, *args, **kwargs):
        """
        Convert the given note data into messages that can be written with mido
        :param note_data: the data to convert
        :param args: args that could be passed to later implementations
        :param kwargs: kwargs that could be passed to later implementations
        :return: the data converted into mido messages based on this implementation
        """
        raise NotImplementedError("to_messages has not been implemented yet...")

    def to_encoded(self, note_data, *args, **kwargs):
        """
        Converts the note data (after being sent through self.from_messges()) into an encoded representation based
            on this implementation
        :param note_data: the data to encode
        :param args: args that could be passed to later implementations
        :param kwargs: kwargs that could be passed to later implementations
        :return: the data encoded as a numpy array
        """
        raise NotImplementedError("to_encoded has not been implemented yet...")

    def from_encoded(self, binary_data, *args, **kwargs):
        """
        Converts the given binary data into a form that can be passed into self.to_messages()
        :param binary_data: the data to convert
        :param args: args that could be passed to later implementations
        :param kwargs: kwargs that could be passed to later implementations
        :return: the data in an intermediary form
        """
        raise NotImplementedError("from_encoded has not been implemented yet...")

    def from_network(self, data):
        """
        Converts the data from output from a network (float values) to values this midi reader can use (usually
            1's, 0's, and -1's)
        Input should only be a single example
        :param data: the data to convert
        :return: data that can be decoded by this reader
        """
        raise NotImplementedError("from_network has not been implemented yet...")


class TickwiseReader(MidiReader):
    """
    Read data as a set of notes that are on and off every tick
    """

    def __init__(self, ticks_per_sec, num_notes=NUM_NOTES, midi_note_offset=MIDI_NOTE_OFFSET):
        super().__init__(ticks_per_sec, num_notes=num_notes, midi_note_offset=midi_note_offset)
        self.reader_type = 'tickwise'

    def from_messages(self, messages, *args, **kwargs):
        _time_as_total_ticks(messages, self.ticks)

        ret = []  # The return list
        curr_state = []  # The current state

        # Go through every tick, changing state if there is a message at that tick
        for i in range(messages[-1].time):
            while messages[0].time == i:
                if messages[0].note in curr_state:
                    curr_state.remove(messages[0].note)
                else:
                    curr_state.append(messages[0].note)
                messages = messages[1:]
            ret.append([n - self.midi_offset for n in curr_state])  # Add a copy of curr_state to ret

        return np.array(ret)

    def to_messages(self, note_data, *args, **kwargs):
        """
        note_data should be a list of lists, where each sublist contains integers corresponding to
        the notes that are currently on
        """
        note_data.append([])  # End with off all notes
        ret = []  # The list of mido messages to return
        curr_state = []  # The current state

        # Go through each time step in note data, find the differences with curr_state, and add
        #   them as a message to ret
        ticks_since_last_update = 0
        for n in note_data:
            n = list(set(n))
            for on in diff_list(n, curr_state):  # All of the notes that need to turn on
                ret.append(Message(type='note_on', note=on + self.midi_offset,
                                   time=ticks_since_last_update / float(self.ticks)))
                curr_state.append(on)
                ticks_since_last_update = 0
            for off in diff_list(curr_state, n):  # All of the notes that need to turn on
                ret.append(Message(type='note_off', note=off + self.midi_offset,
                                   time=ticks_since_last_update / float(self.ticks)))
                curr_state.remove(off)
                ticks_since_last_update = 0

            ticks_since_last_update += 1
        return ret

    def to_encoded(self, note_data, *args, **kwargs):
        """
        Input shape: (length_of_midi_file * ticks_per_second, notes_currently_on)
        The output will be shape: (length_of_midi_file * ticks_per_second, num_notes)
        Where each '1' means that note is currently on, and '0' means that note is currently off for that index's
            current time step in ticks
        """
        return np.array([[1 if i in curr_state else 0 for i in range(self.num_notes)] for curr_state in note_data])

    def from_encoded(self, binary_data, *args, **kwargs):
        """
        Input shape: (length_of_midi_file * ticks_per_second, num_notes)
        The output will be shape: (length_of_midi_file * ticks_per_second, notes_currently_on)
        Where each sublist contains positive integers for what notes should be turned on at that time step
        """
        return [[i for i in range(self.num_notes) if curr_bits[i] == 1] for curr_bits in binary_data]

    def from_network(self, data):
        """
        Only need to convert to 0's and 1's based on data >= 0.5
        """
        return data >= 0.5


class MidiUpdateReader(MidiReader):
    # The list of possible ways to fill in wait_bits in each datapoint, along with the encoding and decoding
    #   functions that do so.
    _wait_bit_types = {'fill': (_wait_bit_fill_enc, _wait_bit_fill_dec),
                       'single': (_wait_bit_single_enc, _wait_bit_single_dec),
                       'binary': (_wait_bit_binary_enc, _wait_bit_binary_dec)}

    def __init__(self, ticks_per_sec, wait_bits, wait_bit_store_type='fill', include_next=True, num_notes=NUM_NOTES,
                 midi_note_offset=MIDI_NOTE_OFFSET):
        """
        A Reader that reads data as a list of updates along with how long to wait until the next update
        :param ticks_per_sec: the number of updates per second
        :param wait_bits: the number of bits to use to determine how many ticks to wait until the next update
            if None, then wait_bits = self.ticks
        :param wait_bit_store_type: how to write the bits that store information on how long to wait until
            the next update. Available types are:
            - 'fill': fill with 1 from index num_notes -> num_notes + wait_time
            - 'single': place a single 1 at index num_notes + wait_time, and leave the rest 0's
            - 'binary': write the bits as a binary representation of the amount of time to wait
        :param include_next: if True, then any updates that happen 0 time steps after the current one are added
            in to the current set of notes (IE: it will be impossible to have a datapoint that has a wait_time of
            0 ticks to the next update)
        :param num_notes: the number of possible notes that could be hit
        :param midi_note_offset: the value on which A0 starts in midi
        """
        super().__init__(ticks_per_sec, num_notes=num_notes, midi_note_offset=midi_note_offset)
        if wait_bit_store_type not in self._wait_bit_types.keys():
            raise ValueError("Unknown wait type: %s" % wait_bit_store_type)
        self._enc_wait_bits, self._dec_wait_bits = self._wait_bit_types[wait_bit_store_type]
        self.include_next = include_next
        if wait_bits is None:
            self.wait_bits = self.ticks
        else:
            self.wait_bits = wait_bits
        self.wait_bit_store_type = wait_bit_store_type
        self.reader_type = 'update'

        self.update_swi = self.updates_per_dp = None

    def get_params(self):
        return add_dicts(super().get_params(),
                         dict(wait_bit_store_type=self.wait_bit_store_type, include_next=self.include_next))

    def from_messages(self, messages, *args, **kwargs):
        """
        Read data as a list of updates.
        Each update is a list that contains positive and negative integers describing what notes are turned on/off,
            and always ends with the amount of wait time in ticks before that update should be executed
        The indexing starts at 1 to allow A0 to turn off correctly (otherwise -0 == 0)
        :param messages: the mido messages
        """
        _time_as_incremental_ticks(messages, self.ticks)

        ret = []
        i = 0
        while i < len(messages):
            # Make empty updates as long as this message needs
            #   Find the extra time, add in the update as a list with a single number (no note updates), then
            #   set the new time on messages[i]
            while self._enc_wait_bits(messages[i].time, self.wait_bits)[1] != 0:
                _, ex_time = self._enc_wait_bits(messages[i].time, self.wait_bits)
                ret.append([messages[i].time - ex_time, ])
                messages[i].time = ex_time

            notes = [(messages[i].note - self.midi_offset + 1) * (-1 if messages[i].type == 'note_off' else 1),
                     messages[i].time]

            # If we are combining note updates
            i += 1
            while self.include_next and i < len(messages) and messages[i].time == 0:
                notes = [(messages[i].note - self.midi_offset + 1) * (
                    -1 if messages[i].type == 'note_off' else 1)] + notes
                i += 1
            ret.append(notes)

        return ret

    def to_messages(self, note_data, *args, **kwargs):
        """
        Note_data should be a list of lists, each containing all of the notes that need to be changed as integers
            (positive if turned on, negative if turned off), and always ending with a positive integer describing
            the number of ticks to wait until the update occurs
        """
        messages = []

        overflow_time = 0  # The time the overflows from empty updates
        for n in note_data:
            if len(n) == 0 or n[-1] < 0:
                continue
            if len(n) == 1:
                overflow_time += n[0]
            else:
                messages.append(Message(note=abs(n[0]) + self.midi_offset - 1,
                                        type="note_on" if n[0] >= 0 else "note_off",
                                        time=(overflow_time + n[-1]) / self.ticks))
                for m in n[1:-1]:  # All of the extra notes if there are any
                    messages.append(Message(note=abs(m) + self.midi_offset - 1,
                                            type="note_on" if m >= 0 else "note_off", time=0))

        return messages

    def to_encoded(self, note_data, *args, **kwargs):
        """
        Input shape: (num_updates_in_midi, 1 + num_updates_notes)
        Output shape: (num_updates_in_midi, num_notes + self.wait_bits)
        Split into two sections:
            [0, 0, ..., 1, 0, 1, ..., 0]
            |   notes   |  wait_bits  |

            There will be self.num_notes bits in the 'notes' section, and self.wait_bits in the 'wait_bits' section.
            A '1' will mean that indexed note is turned on in this update, and a '-1' means it is turned off.
            All '0's means there was no update, just wait more time
            Wait_bits can be encoded in multiple different ways
        """
        ret = []

        for n in note_data:
            # Do the notes
            notes = [0] * self.num_notes
            for m in n[:-1]:
                notes[abs(m) - 1] = 1 if m >= 0 else -1  # Remember it's abs(m) - 1

            # Add in the wait_bits, we know there will be no extra
            ret.append(notes + self._enc_wait_bits(n[-1], self.wait_bits)[0])

        return np.array(ret)

    def from_encoded(self, binary_data, *args, **kwargs):
        """
        Input shape: (num_updates_in_midi, num_notes + self.wait_bits)
        Output shape: (num_updates_in_midi, 1 + num_updates_notes)
        """
        ret = []

        for n in binary_data:
            notes = []

            # Enumerate the updates. If the update_type is not 0, it is either on or off, so add it to notes
            #   Don't forget to add 1 to i before multiplying by on/off
            for i, u_type in enumerate(n[:self.num_notes]):
                if u_type != 0:
                    notes.append(int((i + 1) * u_type))

            # Add in the wait_time
            ret.append(notes + [self._dec_wait_bits(list(n[self.num_notes:]))])

        return ret

    def from_network(self, data):
        """
        The wait_bits should be either 0 or 1, but the note bits can also be -1
        """
        ret = []
        for update in data:
            app = update[:]
            app[app >= 0.5] = 1

            n = app[:-self.wait_bits]
            n[n <= -0.5] = -1
            app[:-self.wait_bits] = n

            app[(app != -1) & (app != 1)] = 0
            ret.append(app)
        return np.array(ret)

    def augment_data(self, data, **kwargs):
        """
        Augments differently for update-based midi data. Now number of updates per datapoint as opposed to number of
            seconds/ticks per datapoint
        :param data: the data to augment, must be a list of multiple encoded datas
        :param kwargs: extra kwargs for this and future implementations of augment_data. Here, can have values:
            - updates_per_dp: the number of updates per data point.
            - update_swi: the sliding_window_increment. Used to augment dataset to get more data via a sliding
                window over the midi with this as its increment
            - padding: if not None - pad the beginning and end with this value before doing sliding window
        """
        self.updates_per_dp = kwargs['updates_per_dp'] if 'updates_per_dp' in kwargs.keys() else 4
        self.update_swi = kwargs['update_swi'] if 'update_swi' in kwargs.keys() else self.updates_per_dp
        self.padding = kwargs['padding'] if 'padding' in kwargs.keys() else None

        if self.update_swi is not None and (self.update_swi < 1 or int(self.update_swi) - self.update_swi != 0):
            raise ValueError("swi must be an integer >= 1")
        if self.updates_per_dp <= 0 or not isinstance(self.updates_per_dp, int):
            raise ValueError("updates_per_dp must be an integer > 0")

        ret = _rolling_window(data[0], self.updates_per_dp, self.update_swi, self.padding)
        for d in data[1:]:
            ret = np.append(ret, _rolling_window(d, self.updates_per_dp, self.update_swi, self.padding), axis=0)
        return ret


class LinearizedUpdateReader(MidiUpdateReader):
    def linearize(self, data):
        """
        Takes data, and linearizes it.
        Input shape: (num_examples, self.updates_per_dp, self.num_notes + self.wait_bits)
        Output shape: (num_examples, self.updates_per_dp * (self.num_notes + self.wait_bits))
        :param data: the data to linearize
        """
        return np.reshape(data, [data.shape[0], -1])

    def unlinearize(self, data):
        """
        Takes data, and unlinearizes it.
        Intput shape: (num_examples, self.updates_per_dp * (self.num_notes + self.wait_bits))
        Output shape: (num_examples, self.updates_per_dp, self.num_notes + self.wait_bits)
        :param data: the data to unlinearize
        """
        return np.reshape(data, [data.shape[0], self.updates_per_dp, -1])

    def get_all_midi_data(self, path=TRAINING_MIDIS_PATH, load=False, save=True, **kwargs):
        return self.linearize(super().get_all_midi_data(path=path, load=load, save=save, **kwargs))

    def from_network(self, data):
        return super().from_network(self.unlinearize(data)[0])


class LinearizedTickwiseReader(TickwiseReader):
    def linearize(self, data):
        """
        Takes data, and linearizes it.
        Input shape: (num_examples, self.updates_per_dp, self.num_notes + self.wait_bits)
        Output shape: (num_examples, self.updates_per_dp * (self.num_notes + self.wait_bits))
        :param data: the data to linearize
        """
        return np.reshape(data, [data.shape[0], -1])

    def unlinearize(self, data):
        """
        Takes data, and unlinearizes it.
        Intput shape: (num_examples, self.secs_per_dp * (self.num_notes + self.wait_bits))
        Output shape: (num_examples, self.secs_per_dp, self.num_notes + self.wait_bits)
        :param data: the data to unlinearize
        """
        return np.reshape(data, [data.shape[0], self.secs_per_dp, -1])

    def get_all_midi_data(self, path=TRAINING_MIDIS_PATH, load=False, save=True, **kwargs):
        return self.linearize(super().get_all_midi_data(path=path, load=load, save=save, **kwargs))

    def from_network(self, data):
        return super().from_network(self.unlinearize(data))


################
# Test Methods #
################


def test_midi_readers():
    tps = 48
    wait_bits = 30

    print("Testing bit encoders/decoders")

    # Test the bit encoding/decoding
    l1 = _wait_bit_fill_enc(7, 10)[0]
    l2 = _wait_bit_fill_enc(34, 103)[0]
    l3 = _wait_bit_fill_dec(l2)
    assert cmp_list(l1, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0], ordered=True)
    assert l3 == 34

    l1 = _wait_bit_single_enc(4, 13)[0]
    l2 = _wait_bit_single_enc(874, 1028)[0]
    l3 = _wait_bit_single_dec(l2)
    assert cmp_list(l1, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ordered=True)
    assert l3 == 874

    l1 = _wait_bit_binary_enc(21, 10)[0]
    l2 = _wait_bit_binary_enc(1627, 34)[0]
    l3 = _wait_bit_binary_dec(l2)
    assert cmp_list(l1, [0, 0, 0, 0, 0, 1, 0, 1, 0, 1], ordered=True)
    assert l3 == 1627

    print("t1")
    for t in range(1000):
        for b in range(t + 1, t + 100):
            l1 = _wait_bit_single_enc(t, b)[0]
            l2 = _wait_bit_single_dec(l1)
            if l2 != t:
                raise ValueError("TEST FAILED SINGLE ENC/DEC")

    print("t2")
    for t in range(1000):
        for b in range(t + 1, t + 100):
            l1 = _wait_bit_fill_enc(t, b)[0]
            l2 = _wait_bit_fill_dec(l1)
            if l2 != t:
                raise ValueError("TEST FAILED FILL ENC/DEC")

    print("t3")
    for t in range(1000):
        import math
        s = math.ceil(math.log2(t + 1)) + 1
        for b in range(s, s + 100):
            l1 = _wait_bit_binary_enc(t, b)[0]
            l2 = _wait_bit_binary_dec(l1)
            if l2 != t:
                raise ValueError("TEST FAILED BINARY ENC/DEC")

    print("Testing readers")

    # Make all of the readers
    mr1 = TickwiseReader(tps)
    enc = mr1.to_encoded(mr1.from_messages(MidiIO.read_midi(os.path.join(TRAINING_MIDIS_PATH, "mazrka01.mid"))))
    print("Writing file: tickwise.mid")
    MidiIO.write_midi(mr1.to_messages(mr1.from_encoded(enc)), "C:/Users/Babynado/Desktop/tickwise.mid")
    for bit_enc in ["fill", "single", "binary"]:
        for inc_ in [True, False]:
            name = "mid_update-%s,%s.mid" % (bit_enc, inc_)
            print("StaRting:", name)
            reader = MidiUpdateReader(tps, wait_bits, wait_bit_store_type=bit_enc, include_next=inc_)
            enc = reader.to_encoded(
                reader.from_messages(MidiIO.read_midi(os.path.join(TRAINING_MIDIS_PATH, "mazrka01.mid"))))
            print("Writing file:", name)
            MidiIO.write_midi(reader.to_messages(reader.from_encoded(enc)), "C:/Users/Babynado/Desktop/%s" % name)

    print("All done with tests!")
