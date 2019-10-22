import os
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import matplotlib.pyplot as pplot
import numpy as np
from core.dataloader import SequenceDataLoader
from core.model import LSTMModel

import seaborn as sns
# force tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
modelfilename = askopenfilename(initialdir="saved_models", title="Select Model File (h5)", filetypes=[("HDF files", "*.h5")])
logfilename = os.path.splitext(modelfilename)[0] + ".log"
configfilename = os.path.splitext(modelfilename)[0] + ".json"

configs = json.load(open(configfilename, 'r'))
data = SequenceDataLoader()
data.create_generator(configs)

image_output_dir = "images"
image_output_prefix = os.path.splitext(modelfilename)[0].split("/")[-1]

model = LSTMModel()
model.load_model(modelfilename)

print(model.model.summary())

# inspect the training curve
df = pd.read_csv(logfilename,header=0)
pplot.figure("Training Curves")
pplot.plot(df.get('loss'), label='Training')
pplot.plot(df.get('val_loss'), label='Validation')
pplot.plot(df.get('lr'),label='Learning Rate')
pplot.yscale('log')
pplot.grid(which='major',axis='both')
pplot.ylabel('Mean-Squared-Error Loss (Standardised)')
pplot.xlabel('Training Epochs')
pplot.legend(loc='lower right')
pplot.show()

test_loss = model.loss_evaluation_gen(data.test)
print("Train Loss: {0:.2f}\tValidation Loss: {1:.2f}\tTest Loss: {2:.2f}".format(df.get("loss").values[-1], df.get("val_loss").values[-1], test_loss))
y, yhat = model.next_step_prediction_gen(data.test)
Y = data.scale_responses.inverse_transform(y)
Yhat = data.scale_responses.inverse_transform(yhat)

pplot.figure('One Step Prediction Curves')
pplot.plot(Y, label="Y", linestyle='', marker=',')
pplot.plot(Yhat, label="Y\hat", linestyle='', marker=',')
pplot.xlabel("Samples")
pplot.ylabel("Power Consumption (W)")
pplot.legend(loc="lower right")
pplot.grid(which='major', axis='both')
pplot.show()
pplot.savefig("{}/{}_{}".format(image_output_dir, image_output_prefix, "OneStepPredictionCurves.png"))

pplot.figure('One Step Residual')
pplot.plot(Y-Yhat)
pplot.xlabel("Samples")
pplot.ylabel("Power Residual (W)")
pplot.grid(which='major', axis='both')
pplot.show()
pplot.savefig("{}/{}_{}".format(image_output_dir, image_output_prefix, "OneStepResidual.png"))

print("Mean Absolute Error: {0:.2f} Watts".format(np.mean(np.abs((Y-Yhat)))))

pplot.figure('Distribution Visualisation')
sns.distplot(Y, rug=True, kde=False, label='Test Data')
sns.distplot(Yhat, rug=True, kde=False, label='Prediction')
pplot.xlabel('Power Consumption (Watts)')
pplot.ylabel('Histogram')
pplot.legend()
pplot.savefig("{}/{}_{}".format(image_output_dir, image_output_prefix, "OneStepDistributionComparison.png"))

pplot.figure('Residual Distribution Visualisation')
sns.distplot(Y-Yhat, label='Test Data')
pplot.locator_params(axis='x', nbins=20)
pplot.xlabel('Power Residual (y-yhat) (W)')
pplot.ylabel('Normalised Frequency')
upperCI = np.percentile(Y-Yhat,99.5,0)
lowerCI = np.percentile(Y-Yhat,0.5,0)
pplot.axvline(lowerCI, color='r', label="99% Range ({0:.2f} Watts)".format((upperCI - lowerCI)[0]))
pplot.axvline(upperCI, color='r')
pplot.xlim([-800,500])
pplot.legend()
pplot.savefig("{}/{}_{}".format(image_output_dir, image_output_prefix, "OneStepResidualDistribution.png"))
