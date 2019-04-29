""" Code to do random sampling and cross validation (probably not very interesting) """

import os
import re
import json
import time

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from model import generate_data, RNNInput, RNNModelConfig, RNNModel, train_model, evaluate_model


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, 'data')


def gather_dumps(data_path):
    pattern = re.compile('(\d+).txt')
    paths_idx = []
    for f in os.listdir(data_path):
        m = pattern.match(f)
        if m is not None:
            path = os.path.join(data_path, f)
            idx = int(m.group(1))
            paths_idx.append((path, idx))

    def dump2row(dump):
        row = {}
        del dump['valid_preds']
        row.update(dump.pop('config'))
        row.update(dump)
        return row

    data = []
    index = []
    for path, idx in paths_idx:
        with open(path, 'r') as f:
            dump_json = json.loads(f.read())
        data.append(dump2row(dump_json))
        index.append(idx)
    res = pd.DataFrame(data=data, index=index).sort_index()

    return res


def verify_valid_loss(fname):
    with open(fname, 'r') as f:
        dump_json = json.loads(f.read())

    valid_preds = np.array(dump_json['valid_preds'])
    return np.mean(np.power(valid_preds[:, 0] - valid_preds[:, 1], 2))



def run_validation(data_path, n_times, num_epochs, train_df, valid_df):

    for i in range(n_times):

        print 'RUN {}'.format(i)

        train_input = RNNInput(train_df)
        config = RNNModelConfig.init_random()
        print(config)

        tf.reset_default_graph()
        with tf.variable_scope("Model", reuse=None):
            model = RNNModel(config)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Train model
            start = time.time()
            try:
                train_losses = train_model(sess, model, train_input, num_epochs)

            except ValueError:
                train_loss = np.nan
                valid_loss = np.nan
                train_time = 0.0
                valid_preds = []

            else:
                train_time = time.time() - start
                # Dump figure
                learning_fig = plt.semilogy(train_losses)[0].figure
                fig_fname = '{}.png'.format(i)
                learning_fig.savefig(os.path.join(data_path, fig_fname))
                plt.close(learning_fig)

                # Evaluate model
                train_loss, _ = evaluate_model(sess, model, train_input)
                valid_input = RNNInput(valid_df)
                valid_loss, valid_preds = evaluate_model(sess, model, valid_input)

                print "Training loss: {}".format(train_loss)
                print
                print "Validation loss: {}".format(valid_loss)
                print

            finally:
                print 'Dumping...'
                # Dump results
                res = {
                    'train_loss': float(train_loss),
                    'train_time': float(train_time),
                    'valid_loss': float(valid_loss),
                    'config': config.as_dict(),
                    'valid_preds': np.array(valid_preds).astype(float).tolist()
                }
                res_fname = '{}.txt'.format(i)
                with open(os.path.join(data_path, res_fname), 'w') as f:
                    json.dump(res, f)

            print 'Done!'
            print


if __name__ == '__main__':

    np.random.seed(1)
    # Data parameters
    train_size = 100000
    valid_size = 10000

    value_low = -100
    value_high = 100
    min_length = 1
    max_length = 10
    num_epochs = 5

    train_df = generate_data(size=train_size, value_low=value_low, value_high=value_high,
                             min_length=min_length, max_length=max_length)
    valid_df = generate_data(size=valid_size, value_low=value_low, value_high=value_high,
                             min_length=min_length, max_length=max_length)

    data_path = os.path.join(DATA_PATH, '2eve')

    train_df.to_csv(os.path.join(data_path, 'train.csv'))
    valid_df.to_csv(os.path.join(data_path, 'valid.csv'))

    run_validation(data_path, 100, num_epochs, train_df, valid_df)
