from AutoEncoders import *
from MidiData import *
import shutil

# Midi update reader stuff
tps = 36
wait_bits = 36
bit_type = 'fill'

# model info
model_type = 'mlp'
model_name = model_type + '_ae'
midi_type = 'tickwise'
delete_dir = True
load = None

if delete_dir and load is None and os.path.exists(os.path.join(W_DIR, model_name)):
    shutil.rmtree(os.path.join(W_DIR, model_name))

reader_kwargs = {
    'secs_per_dp': 4,
    'updates_per_dp': 32,
    'update_swi': 8,
    'swi': 4,
    'load': True,
    'save': True,
}

# RecurrentAE build args
rec_build_kwargs = {
    'in_act': 'relu',
    'rec_act': 'softsign',
    'units': 256,
    'layer_type': 'gru',
    'dropout': 0.0,
    'rec_dropout': 0.0,
    'opt': 'adam',
    'loss': 'binary_crossentropy'
}

# MLP build args
mlp_build_kwargs = {
    'in_act': 'relu',
    'last_act': 'relu',
    'batch_norm': True,
    'dropout': 0.0,
    'opt': 'nadam',
    'loss': 'mse'
}

# MLP build args
conv_build_kwargs = {
    'in_act': 'relu',
    'last_act': 'relu',
    'batch_norm': True,
    'dropout': 0.0,
    'opt': 'nadam',
    'loss': 'binary_crossentropy'
}

training_kwargs = {
    'test_size': 0.1,
    'epochs': 2,
    'batch_size': 64,
    'load': load,
    'model_name': model_name
}

if model_type == 'rec':
    model = RecurrentAutoEncoder(model_name)
    bk = rec_build_kwargs
elif model_type == 'mlp':
    model = AutoEncoder(model_name)
    bk = mlp_build_kwargs
elif model_type == 'conv':
    model = ConvolutionalAutoEncoder(model_name)
    bk = conv_build_kwargs
else:
    raise ValueError("Unknown model_type: %s" % model_type)


if model_type == 'rec' or model_type == 'conv':
    if midi_type == 'update':
        reader = MidiUpdateReader(tps, wait_bits, wait_bit_store_type=bit_type)
    elif midi_type == 'tickwise':
        reader = TickwiseReader(tps)
    else:
        raise ValueError("Unknown midi_type: %s" % midi_type)
elif model_type == 'mlp':
    if midi_type == 'update':
        reader = LinearizedUpdateReader(tps, wait_bits, wait_bit_store_type=bit_type)
    elif midi_type == 'tickwise':
        reader = LinearizedTickwiseReader(tps)
    else:
        raise ValueError("Unknown midi_type: %s" % midi_type)
else:
    raise ValueError("Unknown model_type: %s" % model_type)

model.training_session(reader, reader_kwargs=reader_kwargs, build_kwargs=bk, **training_kwargs)

#model.training_session(reader, iterations=0, reader_kwargs=reader_kwargs, build_kwargs=bk, **training_kwargs)

#import EncodeData
#EncodeData.encode_data(model, reader, "C:/Users/Babynado/Desktop/pres_midis", reader_kwargs)

