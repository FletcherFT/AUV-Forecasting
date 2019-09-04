import json
import os
from core.dataloader import SequenceDataLoader
from core.model import LSTMModel
# force tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# def main():
# Load the configuration
configs = json.load(open('SequenceConfig.json', 'r'))
if not os.path.exists(configs['model']['save_dir']):
    os.makedirs(configs['model']['save_dir'])
if not os.path.exists(configs['training']['log_dir']):
    os.makedirs(configs['model']['log_dir'])

# Load the data sets as TimeSeriesGenerators
data = SequenceDataLoader()
data.create_generator(configs)
# data.create_in_memory_set(configs)

# Init the model
model = LSTMModel()
model.build_model(configs)

# Try to train the model
model.train_generator(data, configs)
# model.train(data,configs)

# if __name__=="__main__":
#     main()
