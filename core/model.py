import datetime as dt
import os
import json
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN, CSVLogger, EarlyStopping
from keras.layers import Dense, Dropout, LSTM, InputLayer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from core.utils import Timer, least_trimmed_absolute_value
from core.clr_callback import CyclicLR
from numpy import concatenate


class LSTMModel:
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for i, layer in enumerate(configs['model']['layers']):
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = configs['data']['sequence_length'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
        optimizer = Adam(lr=configs['model']['lr'])
        self.model.compile(loss=configs['model']['loss'], optimizer=optimizer)

        print('[Model] Model Compiled')
        print(self.model.summary())
        timer.stop()

    def train_generator(self, data_obj, configs):
        timer = Timer()
        timer.start()
        epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']
        save_dir = configs['model']['save_dir']
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        timestamp = dt.datetime.now().strftime('%d%m%Y-%H%M%S')
        model_fname = os.path.join(save_dir, '%s-e%s.h5' % (timestamp, str(epochs)))
        statistics_fname = os.path.join(save_dir, '%s-e%s.log' % (timestamp, str(epochs)))
        configs_fname = os.path.join(save_dir, '%s-e%s.json' % (timestamp, str(epochs)))
        with open(configs_fname, 'w') as f:
            json.dump(configs, f, indent=4)
        callbacks = [
            ModelCheckpoint(filepath=model_fname, monitor='loss', save_best_only=True),
            TensorBoard(log_dir=configs['training']['log_dir'],
                        histogram_freq=0,
                        batch_size=configs['training']['batch_size'],
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None,
                        embeddings_data=None,
                        update_freq='epoch'),
            ReduceLROnPlateau(monitor='loss',
                              factor=0.5,
                              patience=10,
                              min_lr=1e-6,
                              verbose=1,
                              cooldown=0),
            # CyclicLR(base_lr=configs['model']['lr'],
            #          max_lr=0.1,
            #          step_size=8*(data_obj.train_samples-configs['data']['sequence_length'])/configs['training']['batch_size'],
            #          scale_fn=None,
            #          mode='triangular2'),
            TerminateOnNaN(),
            CSVLogger(statistics_fname, separator=',', append=True),
            EarlyStopping(monitor='loss',
                          patience=10,
                          verbose=1
                          )
        ]
        try:
            self.model.fit_generator(
                data_obj.train,
                steps_per_epoch=None,
                epochs=configs['training']['epochs'],
                verbose=0,
                callbacks=callbacks,
                validation_data=data_obj.validation,
                validation_steps=None,
                shuffle=True,
                max_queue_size=100,
                workers=5,
                use_multiprocessing=False
            )
        except KeyboardInterrupt:
            print('[Model] Training Interrupted by Keyboard. Model saved as %s' % model_fname)
            timer.stop()
            return

        print('[Model] Training Completed. Model saved as %s' % model_fname)
        timer.stop()

    def train(self, data_obj, configs):
        timer = Timer()
        timer.start()
        epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']
        save_dir = configs['model']['save_dir']
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        timestamp = dt.datetime.now().strftime('%d%m%Y-%H%M%S')
        model_fname = os.path.join(save_dir, '%s-e%s.h5' % (timestamp, str(epochs)))
        statistics_fname = os.path.join(save_dir, '%s-e%s.log' % (timestamp, str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=model_fname, monitor='loss', save_best_only=True),
            TensorBoard(log_dir=configs['training']['log_dir'],
                        histogram_freq=5,
                        batch_size=configs['training']['batch_size'],
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None,
                        embeddings_data=None,
                        update_freq='epoch'),
            # ReduceLROnPlateau(monitor='val_loss',
            #                   factor=0.2,
            #                   patience=4,
            #                   min_lr=1e-6,
            #                   verbose=1,
            #                   cooldown=0),
            CyclicLR(base_lr=configs['model']['lr'],
                     max_lr=0.1,
                     step_size=8 * (data_obj.train_samples - configs['data']['sequence_length']) / configs['training'][
                         'batch_size'],
                     scale_fn=None,
                     mode='triangular2'),
            TerminateOnNaN(),
            CSVLogger(statistics_fname, separator=',', append=True),
            EarlyStopping(monitor='val_loss',
                          patience=10,
                          verbose=1
                          )
        ]

        self.model.fit(x=data_obj.X_train,
                       y=data_obj.y_train,
                       batch_size=configs['training']['batch_size'],
                       epochs=configs['training']['epochs'],
                       validation_split=configs['data']['validation_portion'],
                       shuffle=True,
                       steps_per_epoch=None,
                       validation_steps=None,
                       callbacks=callbacks
                       )
        print('[Model] Training Completed. Model saved as %s' % model_fname)
        timer.stop()

    def next_step_prediction_gen(self, test_gen):
        y, yhat = list(), list()
        for batch in test_gen:
            X = batch[0]
            y.append(batch[1])
            yhat.append(self.model.predict(X))
        y = concatenate(y, axis=0)
        yhat = concatenate(yhat, axis=0)
        return y, yhat

    def loss_evaluation_gen(self, test_gen):
        loss = self.model.evaluate_generator(test_gen)
        return loss

    def continue_training(self, epoch_start, model_fname, statistics_fname, data_obj, configs):
        timer = Timer()
        timer.start()
        epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']
        print('[Model] Training Continuation')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        callbacks = [
            ModelCheckpoint(filepath=model_fname, monitor='loss', save_best_only=True),
            TensorBoard(log_dir=configs['training']['log_dir'],
                        histogram_freq=0,
                        batch_size=configs['training']['batch_size'],
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None,
                        embeddings_data=None,
                        update_freq='epoch'),
            ReduceLROnPlateau(monitor='loss',
                              factor=0.5,
                              patience=5,
                              min_lr=1e-6,
                              verbose=1,
                              cooldown=0),
            # CyclicLR(base_lr=configs['model']['lr'],
            #          max_lr=0.1,
            #          step_size=8*(data_obj.train_samples-configs['data']['sequence_length'])/configs['training']['batch_size'],
            #          scale_fn=None,
            #          mode='triangular2'),
            TerminateOnNaN(),
            CSVLogger(statistics_fname, separator=',', append=True),
            EarlyStopping(monitor='loss',
                          patience=10,
                          verbose=1
                          )
        ]
        try:
            self.model.fit_generator(
                data_obj.train,
                steps_per_epoch=None,
                epochs=configs['training']['epochs'],
                verbose=0,
                callbacks=callbacks,
                validation_data=data_obj.validation,
                validation_steps=None,
                shuffle=True,
                max_queue_size=100,
                workers=5,
                use_multiprocessing=False,
                initial_epoch=epoch_start
            )
        except KeyboardInterrupt:
            print('[Model] Training Interrupted by Keyboard. Model saved as %s' % model_fname)
            timer.stop()
            return

        print('[Model] Training Completed. Model saved as %s' % model_fname)
        timer.stop()


class DenseModel:
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath, custom_objects=None):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath, custom_objects=custom_objects)

    def build_model(self, configs):
        timer = Timer()
        timer.start()
        input_dim = len(configs['data']['predictors'])
        output_dim = len(configs['data']['responses'])

        for i, layer in enumerate(configs['model']['layers']):
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            if layer['type'] == 'input':
                assert i == 0
                self.model.add(InputLayer(input_shape=(input_dim,)))
            if layer['type'] == 'dense':
                # if the input layer was not given
                if i == 0:
                    self.model.add(Dense(neurons, activation=activation, input_shape=(input_dim,)))
                # if the output layer is reached
                elif i == len(configs['model']['layers']) - 1:
                    self.model.add(Dense(output_dim, activation=activation))
                # if it's an intermediate layer
                else:
                    self.model.add(Dense(neurons, activation=activation))
            # handle dropout layer as input
            if layer['type'] == 'dropout':
                if i == 0:
                    self.model.add(Dropout(dropout_rate, input_shape=(input_dim,)))
                else:
                    self.model.add(Dropout(dropout_rate))
        optimizer = Adam(lr=configs['model']['lr'])
        self.model.compile(loss=least_trimmed_absolute_value(output_dim-1), optimizer=optimizer)

        print('[Model] Model Compiled')
        print(self.model.summary())
        timer.stop()

    def train(self, data_obj, configs):
        timer = Timer()
        timer.start()
        epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']
        save_dir = configs['model']['save_dir']
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        timestamp = dt.datetime.now().strftime('%d%m%Y-%H%M%S')
        model_fname = os.path.join(save_dir, '%s-e%s.h5' % (timestamp, str(epochs)))
        statistics_fname = os.path.join(save_dir, '%s-e%s.log' % (timestamp, str(epochs)))
        configs_fname = os.path.join(save_dir, '%s-e%s.json' % (timestamp, str(epochs)))
        with open(configs_fname, 'w') as f:
            json.dump(configs, f, indent=4)
        callbacks = [
            ModelCheckpoint(filepath=model_fname, monitor='loss', save_best_only=True),
            TensorBoard(log_dir=configs['training']['log_dir'],
                        histogram_freq=10,
                        batch_size=configs['training']['batch_size'],
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None,
                        embeddings_data=None,
                        update_freq='epoch'),
            ReduceLROnPlateau(monitor='loss',
                              factor=0.5,
                              patience=5,
                              min_lr=1e-6,
                              verbose=1,
                              cooldown=0),
            TerminateOnNaN(),
            CSVLogger(statistics_fname, separator=',', append=True),
            EarlyStopping(monitor='loss',
                          patience=10,
                          verbose=1
                          )
        ]
        try:
            self.model.fit(x=data_obj.train[0],
                           y=data_obj.train[1],
                           batch_size=configs['training']['batch_size'],
                           epochs=configs['training']['epochs'],
                           validation_data=data_obj.validation,
                           shuffle=True,
                           steps_per_epoch=None,
                           validation_steps=None,
                           callbacks=callbacks
                           )
        except KeyboardInterrupt:
            print('[Model] Training Interrupted by Keyboard. Model saved as %s' % model_fname)
            timer.stop()
            return

        print('[Model] Training Completed. Model saved as %s' % model_fname)
        timer.stop()

    def continue_training(self, epoch_start, model_fname, statistics_fname, data_obj, configs):
        timer = Timer()
        timer.start()
        epochs = configs['training']['epochs']
        batch_size = configs['training']['batch_size']
        print('[Model] Training Continuation')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        callbacks = [
            ModelCheckpoint(filepath=model_fname, monitor='loss', save_best_only=True),
            TensorBoard(log_dir=configs['training']['log_dir'],
                        histogram_freq=5,
                        batch_size=configs['training']['batch_size'],
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None,
                        embeddings_data=None,
                        update_freq='epoch'),
            ReduceLROnPlateau(monitor='loss',
                              factor=0.5,
                              patience=5,
                              min_lr=1e-6,
                              verbose=1,
                              cooldown=0),
            TerminateOnNaN(),
            CSVLogger(statistics_fname, separator=',', append=True),
            EarlyStopping(monitor='loss',
                          patience=10,
                          verbose=1,
                          min_delta=1e-8,
                          )
        ]
        try:
            self.model.fit(x=data_obj.train[0],
                           y=data_obj.train[1],
                           batch_size=configs['training']['batch_size'],
                           epochs=configs['training']['epochs'],
                           validation_data=data_obj.validation,
                           shuffle=True,
                           steps_per_epoch=None,
                           validation_steps=None,
                           callbacks=callbacks,
                           initial_epoch=epoch_start
                           )
        except KeyboardInterrupt:
            print('[Model] Training Interrupted by Keyboard. Model saved as %s' % model_fname)
            timer.stop()
            return

        print('[Model] Training Completed. Model saved as %s' % model_fname)
        timer.stop()
