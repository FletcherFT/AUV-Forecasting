import glob
import os
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from core.utils import Timer
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats


def series_to_supervised(df, configs, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        df: Sequence of observations a pandas dataframe.
        configs: dictionary of data configuration.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        X: Pandas DataFrame of series framed for supervised learning (predictor)..
        y: Pandas DataFrame of series framed for supervised learning (response).
    """
    predictors = df.get(configs['data']['predictors'])
    responses = df.get(configs['data']['responses'])
    cols, predictor_names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(predictors.shift(i))
        predictor_names += [('%s(t-%d)' % (j, i)) for j in configs['data']['predictors']]
    response_names = list()
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(responses.shift(-i))
        if i == 0:
            response_names += [('%s(t)' % j) for j in configs['data']['responses']]
        else:
            response_names += [('%s(t+%d)' % (j, i)) for j in configs['data']['responses']]
    agg = pd.concat(cols, axis=1)
    agg.columns = predictor_names + response_names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    X = agg.get(predictor_names).values
    y = agg.get(response_names).values
    return X, y


class SequenceDataLoader:
    def __init__(self):
        self.train = None
        self.validation = None
        self.test = None
        self.scale = None
        self.scale_predictors = None
        self.scale_responses = None
        self.train_samples = 0

    def create_generator(self, configs):
        # Start a timer
        timer = Timer()
        timer.start()
        # Get all the csv files specified by configs
        l = list()
        files = glob.glob(os.path.join(configs['data']['directory'], configs['data']['filename']))
        # Concatenate all the data frames into one huge sequence
        for f in files:
            df = pd.read_csv(f)
            l.append(df)
        df = pd.concat(l)
        self.scale = StandardScaler().fit(df.get(configs['data']['predictors'] + configs['data']['responses']))
        self.scale_predictors = StandardScaler().fit(df.get(configs['data']['predictors']))
        self.scale_responses = StandardScaler().fit(df.get(configs['data']['responses']))
        df = df.dropna()
        X = df.get(configs['data']['predictors']).values
        y = df.get(configs['data']['responses']).values
        X_train, X_remain, y_train, y_remain = train_test_split(X, y,
                                                                test_size=configs['data']['validation_portion'] +
                                                                          configs['data']['test_portion']
                                                                , train_size=configs['data']['train_portion'],
                                                                shuffle=False)
        X_validation, X_test, y_validation, y_test = train_test_split(X_remain, y_remain,
                                                                      test_size=configs['data']['test_portion'] / (
                                                                              configs['data'][
                                                                                  'validation_portion'] +
                                                                              configs['data']['test_portion']),
                                                                      train_size=configs['data'][
                                                                                     'validation_portion'] / (
                                                                                         configs['data'][
                                                                                             'validation_portion'] +
                                                                                         configs['data'][
                                                                                             'test_portion']),
                                                                      shuffle=False)
        self.train_samples = X_train.shape[0]
        self.train = TimeseriesGenerator(self.scale_predictors.transform(X_train),
                                         self.scale_responses.transform(y_train),
                                         length=configs['data']['sequence_length'],
                                         sampling_rate=1,
                                         stride=1,
                                         start_index=0,
                                         end_index=None,
                                         shuffle=False,
                                         reverse=False,
                                         batch_size=configs['training']['batch_size'])
        self.validation = TimeseriesGenerator(self.scale_predictors.transform(X_validation),
                                              self.scale_responses.transform(y_validation),
                                              length=configs['data']['sequence_length'],
                                              sampling_rate=1,
                                              stride=1,
                                              start_index=0,
                                              end_index=None,
                                              shuffle=False,
                                              reverse=False,
                                              batch_size=configs['training']['batch_size'])
        self.test = TimeseriesGenerator(self.scale_predictors.transform(X_test),
                                        self.scale_responses.transform(y_test),
                                        length=configs['data']['sequence_length'],
                                        sampling_rate=1,
                                        stride=1,
                                        start_index=0,
                                        end_index=None,
                                        shuffle=False,
                                        reverse=False,
                                        batch_size=configs['training']['batch_size']
                                        )
        print('[Data] Data loaded and ready for training.')
        timer.stop()

    def create_batches(self, configs):
        timer = Timer()
        timer.start()
        l = list()
        # concatenate all the data files so that scalers can be constructed
        files = glob.glob(os.path.join(configs['data']['directory'], configs['data']['filename']))
        for f in files:
            df = pd.read_csv(f)
            l.append(df)
        df = pd.concat(l)
        self.scale = StandardScaler().fit(df.get(configs['data']['predictors'] + configs['data']['responses']))
        self.scale_predictors = StandardScaler().fit(df.get(configs['data']['predictors']))
        self.scale_responses = StandardScaler().fit(df.get(configs['data']['responses']))
        num_X_features = len(df.get(configs['data']['predictors']).columns)
        num_X_timesteps = configs['data']['sequence_length']
        X = np.empty((0, num_X_timesteps, num_X_features), float)
        num_y_features = len(df.get(configs['data']['responses']).columns)
        num_y_timesteps = configs['data']['forecast_length']
        y = np.empty((0, num_y_timesteps, num_y_features), float)
        # go through each dataframe, create supervised datasets and append to X and y
        for df in l:
            tmpX, tmpy = series_to_supervised(df, configs, num_X_timesteps, num_y_timesteps)
            X = np.append(X, np.reshape(tmpX, (tmpX.shape[0], num_X_timesteps, num_X_features)))
            y = np.append(y, np.reshape(tmpy, (tmpy.shape[0], num_y_timesteps, num_y_features)))
        # separate X, y into training and testing sets
        X_train, X_remain, y_train, y_remain = train_test_split(X, y,
                                                                test_size=configs['data']['validation_portion'] +
                                                                          configs['data']['test_portion']
                                                                , train_size=configs['data']['train_portion'],
                                                                shuffle=False)
        X_validation, X_test, y_validation, y_test = train_test_split(X_remain, y_remain,
                                                                      test_size=configs['data']['test_portion'] / (
                                                                              configs['data']['validation_portion'] +
                                                                              configs['data']['test_portion']),
                                                                      train_size=configs['data']['validation_portion'] / (
                                                                                 configs['data']['validation_portion'] +
                                                                                 configs['data'][
                                                                                     'test_portion']),
                                                                      shuffle=False)
        self.train = (X_train, y_train)
        self.validation = (X_validation, y_validation)
        self.test = (X_test, y_test)

        print('[Data] Data loaded and ready for training.')
        timer.stop()


class StatelessDataLoader:
    def __init__(self):
        self.data_frame = None
        self.train = None
        self.validation = None
        self.test = None
        self.predictor_scaler = None
        self.response_scaler = None

    def create_in_memory(self, configs, reproducible=True):
        # Start a timer
        timer = Timer()
        timer.start()
        # Get all the csv files specified by configs
        tmp = list()
        files = glob.glob(os.path.join(configs['data']['directory'], configs['data']['filename']))
        # Concatenate all the data frames into one data frame
        for f in files:
            df = pd.read_csv(f)
            tmp.append(df)
        df = pd.concat(tmp)
        self.data_frame = df
        # Clean the data by dropping NaNs
        df = df.dropna()
        # # Remove ridiculous outliers from the data
        # tmp = df.get([
        #     "longitudinal_velocity_ins",
        #     "transverse_velocity_ins",
        #     "vertical_velocity_ins",
        #     "roll_rate",
        #     "pitch_rate",
        #     "yaw_rate"])
        # # remove based on the interquartile range
        # Q1 = tmp.quantile(0.25)
        # Q3 = tmp.quantile(0.75)
        # IQR = Q3-Q1
        # tmp = tmp.where(~((tmp < (Q1 - 1.5 * IQR)) | (tmp > (Q3 + 1.5 * IQR))).any(axis=1))
        # tmp = tmp.interpolate()
        # df[
        #     "longitudinal_velocity_ins",
        #     "transverse_velocity_ins",
        #     "vertical_velocity_ins",
        #     "roll_rate",
        #     "pitch_rate",
        #     "yaw_rate"] = tmp

        # Fit standardised scalers to the data
        self.predictor_scaler = StandardScaler().fit(df.get(configs['data']['predictors']))
        self.response_scaler = StandardScaler().fit(df.get(configs['data']['responses']))
        # Separate into predictor/response values
        X = df.get(configs['data']['predictors']).values
        y = df.get(configs['data']['responses']).values
        # Scale the data if specified
        if configs['data']['normalise']:
            X = self.predictor_scaler.transform(X)
            y = self.response_scaler.transform(y)

        # Split the training data, optional to maintain reproducibility (useful for k-folds test)
        X_train, X_remain, y_train, y_remain = train_test_split(X, y,
                                                                test_size=configs['data']['validation_portion'] +
                                                                          configs['data']['test_portion']
                                                                , train_size=configs['data']['train_portion'],
                                                                shuffle=reproducible,
                                                                random_state=42)
        X_validation, X_test, y_validation, y_test = train_test_split(X_remain, y_remain,
                                                                      test_size=configs['data']['test_portion'] / (
                                                                              configs['data'][
                                                                                  'validation_portion'] +
                                                                              configs['data']['test_portion']),
                                                                      train_size=configs['data'][
                                                                                     'validation_portion'] / (
                                                                                         configs['data'][
                                                                                             'validation_portion'] +
                                                                                         configs['data'][
                                                                                             'test_portion']),
                                                                      shuffle=reproducible,
                                                                      random_state=42)

        self.train = (X_train, y_train)
        self.validation = (X_validation, y_validation)
        self.test = (X_test, y_test)
        print('[Data] Data loaded and ready for training.')
        timer.stop()
