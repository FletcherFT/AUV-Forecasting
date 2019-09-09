import os
import sys
import json
import pandas as pd
import glob
import difflib
from core.dataloader import StatelessDataLoader
from core.model import DenseModel

print("Available Targets:")
dirlist = glob.glob("saved_models/*.h5")
separator = "\n"
print(separator.join(dirlist))
enteredname = input("Enter name or part of name from above:")
bestchoices = difflib.get_close_matches(enteredname,dirlist)
assert len(bestchoices)>0
modelfilename = bestchoices[0]
print("File: '{}' Selected.".format(modelfilename))
logfilename = os.path.splitext(modelfilename)[0] + ".log"
configfilename = os.path.splitext(modelfilename)[0] + ".json"

df = pd.read_csv(logfilename,header=0)
epoch_start = len(df.epoch)

configs = json.load(open(configfilename, 'r'))

data = StatelessDataLoader()
data.create_in_memory(configs)

model = DenseModel()
model.load_model(modelfilename)

model.continue_training(epoch_start, modelfilename, logfilename, data, configs)
