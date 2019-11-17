from mido import MidiFile, Message, MidiTrack, second2tick


def write_midi(messages, file):
    """
    Writes the list of midi messages to a midi file
    :param messages: the midi data to write
    :param file: the string of full path+filename to write to
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for message in messages:
        message.time = int(round(second2tick(message.time, mid.ticks_per_beat, 500000)))
        track.append(message)

    mid.save(file)


def read_midi(path):
    """
    Reads in the note data from the given midi file. Returns all note_on and note_off messages
    :param path: the path to the midi
    :return: all note messages
    """
    messages = []
    elapsed = 0
    for msg in MidiFile(path):
        if not msg.is_meta:
            if msg.type in ['note_on', 'note_off']:
                msg.time += elapsed
                messages.append(msg)
                elapsed = 0
            else:
                elapsed += msg.time
    return messages
