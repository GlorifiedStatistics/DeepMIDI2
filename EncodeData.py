from MidiData import *
import shutil

def encode_data(model, reader, pres_path, reader_kwargs):
    """
    Make examples for use in presentation, as well as encode data for RNN
    :param model: the model to use to encode
    :param reader: the midi reader
    :param pres_path: the path to make presentation data
    :param reader_kwargs: kwargs for the data reader
    """
    if os.path.exists(pres_path):
        shutil.rmtree(pres_path)
    os.mkdir(pres_path)

    tick_inc = reader.ticks * reader_kwargs['secs_per_dp'] \
        if isinstance(reader, TickwiseReader) else reader_kwargs['updates_per_dp']

    # Write the presentation midi
    data = reader.to_encoded(reader.from_messages(MidiIO.read_midi(os.path.join(TRAINING_MIDIS_PATH, 'mazrka03.mid'))))
    encoded = None

    i = 0
    while i < len(data)/2:
        d = np.array([data[i:i+tick_inc]])
        d = np.reshape(d, list(d.shape) + [1])
        if encoded is None:
            encoded = np.reshape(model.model.predict(d), d.shape[1:3])
        else:
            encoded = np.append(encoded, np.reshape(model.model.predict(d), d.shape[1:3]), axis=0)
        i += tick_inc

    out = reader.to_messages(reader.from_encoded(reader.from_network(encoded)))
    MidiIO.write_midi(out, os.path.join(pres_path, "pres_mid.mid"))
