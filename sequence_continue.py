import os
import json
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from core.dataloader import SequenceDataLoader
from core.model import LSTMModel

# force tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
modelfilename = askopenfilename(initialdir="saved_models", title="Select Model File (h5)", filetypes=[("HDF files", "*.h5")])
logfilename = os.path.splitext(modelfilename)[0] + ".log"
configfilename = os.path.splitext(modelfilename)[0] + ".json"

df = pd.read_csv(logfilename,header=0)
epoch_start = len(df.epoch)

configs = json.load(open(configfilename, 'r'))

data = SequenceDataLoader()
data.create_generator(configs)

model = LSTMModel()
model.load_model(modelfilename)

model.continue_training(epoch_start, modelfilename, logfilename, data, configs)