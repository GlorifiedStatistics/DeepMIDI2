from MidiModels import *

ticks = 36
num_secs = 1
notes = 88
sliding_window_inc = 8

curr_model_name = "model_conv"

load = False
version = 'latest'

if load:
    model = load_model(curr_model_name, version=version, sliding_window_inc=sliding_window_inc)
    print(model.model.summary())
    print("Starting loaded model on iteration:", model.iteration)
else:
    model = AutoEncoder(curr_model_name, (ticks, num_secs, sliding_window_inc))
    model.build_model()
    print(model.model.summary())

model.train_model(2000, batch_size=256)

