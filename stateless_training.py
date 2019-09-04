import json
import os
from core.dataloader import StatelessDataLoader
from core.model import DenseModel
# force tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# def main():
# Load the configuration
configs = json.load(open('StatelessConfig.json', 'r'))
if not os.path.exists(configs['model']['save_dir']):
    os.makedirs(configs['model']['save_dir'])
if not os.path.exists(configs['training']['log_dir']):
    os.makedirs(configs['model']['log_dir'])

# Load the data sets as TimeSeriesGenerators
data = StatelessDataLoader()
data.create_in_memory(configs)

# Init the model
model = DenseModel()
model.build_model(configs)

# Try to train the model
model.train(data, configs)
