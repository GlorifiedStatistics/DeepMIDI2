from Constants import *
from PythonUtils import *
from mido import Message
import MidiIO
import pickle


def encode_notes(note_data, ticks):
    """
    Encode the given note data into data the network can use. Time is split into 1/ticks seconds for each tick. Each
        tick contains what notes should be held on during that tick. Notes are held between ticks if the next tick
        contains that note as well.
    :param note_data: the note date
    :param ticks: the number of ticks per second
    :return: encoded note data
    """
    ret = []
    curr = []  # The current list of held notes

    for n in note_data:
        # Stay in the current position until the next note change
        for i in range(max(1, int(n.time / (1.0 / ticks)))):
            ret.append([c for c in curr])

        # Update the curr
        if n.type == "note_on":
            if n.note not in curr:
                curr.append(n.note)
        else:
            if n.note in curr:
                curr.remove(n.note)

    return ret


def decode_notes(encoded_notes, ticks):
    """
    Decodes the given encoded notes into a list of messages that can eventually be written to file
    :param encoded_notes: the encoded notes
    :param ticks: the number of ticks per second
    :return: a list of mido messages
    """
    ret = []
    curr = []  # The currently pressed notes
    t = 0  # The number of ticks since last update

    for en in encoded_notes:
        # Increment time first
        t += 1

        # Do nothing if the lists do not change
        if cmp_list(en, curr):
            continue

        # Find the differences in each list. Those in en but not in curr should now be turned on. Those in curr but
        #   not in en should be turned off.
        on, off = diff_list(en, curr), diff_list(curr, en)
        for o in on:
            ret.append(Message('note_on', note=o, time=t / float(ticks)))
            curr.append(o)
            t = 0
        for o in off:
            ret.append(Message('note_off', note=o, time=t / float(ticks)))
            curr.remove(o)
            t = 0
    if len(curr) > 0:
        ret.append(Message('note_off', note=curr[0], time=(t + 1) / float(ticks)))
    for n in curr[1:]:
        ret.append(Message('note_off', note=n, time=0))
    return ret


def get_autoencoder_data(path, ticks, num_secs, sliding_window_inc):
    """
    Does get_autoencoder_data for all data in dir (all midi files)
    """
    total_data = []
    for file in os.listdir(path):
        if file[-4:] == '.mid':
            total_data += _get_singular_autoencoder_data(os.path.join(path, file), ticks, num_secs, sliding_window_inc)

    return total_data


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
    data = encode_notes(MidiIO.get_note_data(file), ticks)

    curr_idx = 0
    v = int(ticks * num_secs)
    while curr_idx < len(data) - v:
        ret.append(compress_data(data[curr_idx:curr_idx + v]))
        curr_idx += sliding_window_inc if sliding_window_inc is not None else v

    # Also do the last one just cause
    ret.append(compress_data(data[-v:]))

    return ret


def compress_data(data):
    """
    Takes the data (a list of lists of notes that are currently on) and converts into a 1D array of 1's and 0's.
    """
    ret = [0 for i in range(len(data) * 88)]
    for i in range(len(data)):
        for n in data[i]:
            ret[i * 88 + n - MIDI_NOTE_OFFSET] = 1
    return ret


def uncompress_data(data):
    """
    Takes the compressed data and converts back to data that can go through decode_notes
    """
    ret = []
    for i in range(int(len(data) / 88)):
        add = []
        for j in range(88):
            if data[i * 88 + j] > 0.5:
                add.append(j + MIDI_NOTE_OFFSET)
        ret.append(add)
    return ret


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


def test_midi(ticks, num_secs):
    from mido import MidiFile, MidiTrack, second2tick

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for n in decode_notes(encode_notes(MidiIO.get_note_data('mazurka_midis/mazrka10.mid'), ticks), ticks):
        n.time = int(round(second2tick(n.time, mid.ticks_per_beat, 500000)))
        track.append(n)

    mid.save("C:/Users/Babynado/Desktop/test.mid")

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for n in decode_notes(uncompress_data(
            get_autoencoder_data('./mazurka_midis/mazrka01.mid', ticks, num_secs, None)[0]), ticks):
        n.time = int(round(second2tick(n.time, mid.ticks_per_beat, 500000)))
        track.append(n)

    mid.save("C:/Users/Babynado/Desktop/test2.mid")

