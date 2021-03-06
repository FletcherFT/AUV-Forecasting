import datetime as dt
import tensorflow as tf


def least_trimmed_absolute_value(h):
    def inner_loop(y_true, y_pred):
        Err = tf.sort(tf.abs(y_pred-y_true),
               axis=1,
               direction='ASCENDING')
        return tf.reduce_sum(Err[:, 0:h], axis=1)
    return inner_loop


class Timer:

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))
