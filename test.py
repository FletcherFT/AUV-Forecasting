import os
import sys
import json
import pandas as pd
import glob
import difflib
from core.dataloader import StatelessDataLoader
from core.model import DenseModel

model = DenseModel()
configs = json.load(open("./saved_models/05092019-113240-e1000.json", 'r'))
model.build_model(configs)
model.load_model("./saved_models/05092019-113240-e1000.h5")

print(model.summary())

