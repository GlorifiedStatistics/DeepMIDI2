from Constants import *
from mido import Message
import MidiIO
import pickle
import numpy as np


def encode_notes(note_data, ticks, wait_bits=None):
    """
    Encode the given note data into data the network can use. The list of note data (detailing on and off times for
        singular notes) is converted into a single list of 1's and 0's (and occasionally -1's).
        Within each tick, we have:
            [1/0/-1, 0, 0, ..., 1, 0, 1, ..., 0]
            |     |                           |
             type            notes
        Where "type" is:
            1 - if the list of notes is turned on
            0 - if the list of notes is turned off
            -1 - if nothing happens this tick (IE: the following list of notes is irrelevant)
        And "notes" is:
            a list of 88 bits with "1" indicating the note at that index is changed, and "0" meaning it isn't
    :param note_data: the note data
    :param ticks: the number of ticks per second
    :param wait_bits: the number of bits to use to for wait times. If the network needs to wait longer than this time,
        a list of all 0's for the notes will be added repeatedly.
        If None: use 'ticks' bits
    :return: encoded note data
    """
    if wait_bits is None:
        wait_bits = ticks
    inc = 1.0 / ticks

    # Change time to be exact instead of incremental
    total_time = 0
    for n in note_data:
        total_time += n.time
        n.time = total_time

    # Divide by total_time and round to get tick index
    data = []
    for n in note_data:
        n.time = round(n.time / inc)
        data.append([int(n.type == 'note_on'), n.note, n.time])

    # Group notes happening at the same time together
    messages = []
    i = 0
    t = 0  # Keep track of current time
    while i < len(data):
        r = [data[i][1] * (1 if data[i][0] == 1 else -1), data[i][2] - t]
        t = data[i][2]
        while i < len(data) - 1 and data[i + 1][2] == t:
            i += 1
            r = [data[i][1] * (1 if data[i][0] == 1 else -1), ] + r
        messages.append(r)
        i += 1

    ret = []
    for m in messages:
        a = np.zeros([NUM_NOTES + wait_bits, ])  # The zeros
        t = min(m[-1], wait_bits)  # The wait time
        m[-1] -= t
        a[NUM_NOTES:NUM_NOTES + t - 1] = 1  # Set the wait time bit to 1

        # Add all the notes
        for note in m[:-1]:
            a[abs(note) - MIDI_NOTE_OFFSET] = 1 if note > 0 else -1
        ret.append(a)

        # Add all the filler space
        while m[-1] > 0:
            a = np.zeros([NUM_NOTES + wait_bits, ])  # The zeros
            t = min(m[-1], wait_bits)  # The wait time
            m[-1] -= t
            a[NUM_NOTES + t - 1] = 1  # Set the wait time bit to 1
            ret.append(a)

    return np.array(ret)


def decode_notes(encoded_notes, ticks):
    """
    Decodes the given encoded notes into a list of messages that can eventually be written to file
    :param encoded_notes: the encoded notes
    :param ticks: the number of ticks per second
    :return: a list of mido messages
    """
    ret = []
    add_time = 0
    for row in encoded_notes:
        on_notes = np.where(row[:NUM_NOTES] > 0.5)[0]
        off_notes = np.where(row[:NUM_NOTES] < -0.5)[0]
        total_time = len(np.where(row[NUM_NOTES:] > 0.5)[0])
        time = total_time + 1 + add_time if total_time > 0 else add_time + 1
        for n in on_notes:
            ret.append(Message('note_on', note=n + MIDI_NOTE_OFFSET, time=time / float(ticks) if time > 0 else 0))
            time = 0
            add_time = 0
        for n in off_notes:
            ret.append(Message('note_off', note=n + MIDI_NOTE_OFFSET, time=time / float(ticks) if time > 0 else 0))
            time = 0
            add_time = 0
        if len(on_notes) == 0 and len(off_notes) == 0:
            add_time += total_time + 1

    return ret


def get_autoencoder_data(path, ticks, num_secs, sliding_window_inc):
    """
    Does get_autoencoder_data for all data in dir (all midi files)
    """
    total_data = None
    for file in os.listdir(path):
        if file[-4:] == '.mid':
            add = _get_singular_autoencoder_data(os.path.join(path, file), ticks, num_secs, sliding_window_inc)
            if total_data is None:
                total_data = add
            total_data = np.append(total_data, add, axis=0)

    return np.array(total_data)


def _get_singular_autoencoder_data(file, ticks, num_secs, sliding_window_inc):
    """
    Returns all of the data from the given midi file in a format suitable for an autoencoder
    Returns as a 4-tuple of x_train, y_train, x_test, y_test
    :param ticks: the number of ticks per second
    :param num_secs: the number of seconds to include in each example
    :param sliding_window_inc: the number of ticks to shift the sliding window for each new example. If None, then a
        sliding window is not used, and instead the examples are taken one after another as long as time permits
    """
    if sliding_window_inc is not None and sliding_window_inc < 1:
        raise ValueError("Sliding_window_inc must be >= 1, instead is: %d" % sliding_window_inc)

    ret = []
    data = encode_notes(MidiIO.read_midi(file), ticks)

    curr_idx = 0
    v = int(ticks * num_secs)
    while curr_idx < len(data) - v:
        ret.append(data[curr_idx:curr_idx + v])
        curr_idx += sliding_window_inc if sliding_window_inc is not None else v

    # Also do the last one just cause
    ret.append(data[-v:])
    return np.array(ret)


def load_midi_data(path, ticks, num_secs, sliding_window_inc):
    """
    Attempts to load this data if it has already been computed based on a file whose name is a hash of the inputs. If
        that file does not exist in path, then the new midi data is created and saved to a file, then returned.
    :param path: the path to the midi files
    :param ticks: the number of time steps per seconds
    :param num_secs: the number of seconds per data point
    :param sliding_window_inc: if an integer - use a sliding window over the song, incrementing by sliding_window_inc
        for each new data point
        - if None - do not use a sliding window, just split the song up into ticks * num_secs chunks and drop the last
            one if it is too small
    :return: the midi data
    """
    h = str((hash((ticks, num_secs, sliding_window_inc)) + hash(123456789)) % 10**10) + '.pkl'

    print("Loading MIDI data...")
    if h not in os.listdir(path):
        print("Could not find data file, making a new one...")
        data = get_autoencoder_data(path, ticks, num_secs, sliding_window_inc)
        with open(os.path.join(path, h), 'wb') as f:
            pickle.dump(data, f)
        print("Created midi data file: %s" % h)
        return data
    with open(os.path.join(path, h), 'rb') as f:
        return pickle.load(f)
