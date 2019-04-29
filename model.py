""" Code to generate data and train and evaluate the model. """

import itertools

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.python.ops.rnn_cell_impl import BasicRNNCell, _Linear
from tensorflow.python.ops.init_ops import Initializer, \
    Identity, glorot_uniform_initializer


ALPHABET = '0123456789,-_'
NUM_CLASSES = len(ALPHABET)
CHAR2IX = {ch: i for i, ch in enumerate(ALPHABET)}
IX2CHAR = {i: ch for i, ch in enumerate(ALPHABET)}
PADIX = CHAR2IX['_']


def l1norm(x):
    """ Compute the L1-norm of vectors. """
    return np.sqrt(np.sum(np.abs(x), axis=1))


def generate_data(size, value_low=-100, value_high=100, min_length=1, max_length=10):
    """
    Generate examples of random vectors and compute their L1 norm.
    Convert the vectors to strings and encode them with numbers.

    The vectors' elements are drawn uniformly from [`value_low`, `value_high`]
    The vectors' lenghts are chozen uniformly from [`min_length`, `max_length`]

    Parameters
    ----------
    size : int
        Number of examples of vectors to generate
    value_low : int
        Lowest (signed) element to be drawn
    value_high : int
        Largest (signed) element to be drawn
    min_length : int
        Lowest length of a vector
    max_length : int
        Largest length of a vector

    Return
    ------
    DataFrame
        columns: as_string, vector_length, l1_norm, as_numbers, seq_length
    """

    assert min_length > 0
    assert max_length > 0

    # Sample elements and lengths
    x = np.random.randint(value_low, value_high + 1, size=(size, max_length))
    lengths = np.random.randint(min_length, max_length + 1, size=(size,))

    # Compute L1 norms
    for i in range(size):
        x[i, lengths[i]:] = 0
    norms = l1norm(x)

    # Convert the vectors to strings
    # e.g. [12, 1, -98] -> "12,1,-98"
    x_as_str = []
    for i in range(size):
        vec_str = ','.join(str(_) for _ in x[i, :lengths[i]])
        x_as_str.append(vec_str)

    # Create a DataFrame
    data = {
        "as_string": x_as_str,
        "vector_length": lengths,
        "l1norm": norms
    }
    df = pd.DataFrame(data=data)

    # Encode vector string representations with numbers
    # e.g [12, 1, -98] -> [1, 2, 10, 1, 10, 11, 9, 8]
    def int_encode(vec):
        return [CHAR2IX[ch] for ch in vec]

    df["as_numbers"] = df["as_string"].apply(int_encode)
    df["seq_length"] = df["as_numbers"].apply(len)  # Lengths of the encoded sequences

    return df


def exp_uniform_abs(a,b):
    """ E[abs(X)] where X ~ Uniform(a,b) """
    assert b >= a
    if a < 0 < b:
        exp = (a**2 + b**2)*1.0/(b - a)/2
    else:
        exp = (abs(a) + abs(b))*1.0/2
    return exp


def baseline_mse(y, value_low, value_high, min_length, max_length):
    """
    Compute the Mean Squared Error between the ground truths and a baseline prediction.
    As the baseline prediction the expected L1 norm over the specified distribution is taken.
    """
    elem_exp = exp_uniform_abs(value_low, value_high)
    length_exp = exp_uniform_abs(min_length, max_length)  # Length always > 0
    norm_exp = np.sqrt(elem_exp*length_exp)
    return np.mean(np.power(y - norm_exp, 2))


class RNNInput(object):
    """
    Iterator over the input DataFrame.

    The idea is borrowed from:
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
    """

    def __init__(self, df):
        self.df = df.copy()
        self.size = df.shape[0]

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def gen_batch(self, batch_size):
        """
        Return a "batch generator" over the input DataFrame.
        The generator returns batches of inputs along with their targets and
        the length of each input.
        """
        num_complete_batches = self.size // batch_size  # Ignore incomplete batches
        for i in range(num_complete_batches):
            batch_df = self.df.loc[i * batch_size:(i + 1) * batch_size - 1].reset_index(drop=True)
            # Pad sequences so that they are all the same length
            max_length = batch_df['seq_length'].max()
            batch_x = np.ones((batch_size, max_length), dtype=np.int8) * PADIX
            for i in range(batch_size):
                batch_x[i, :batch_df.loc[i, 'seq_length']] = batch_df.loc[i, 'as_numbers']

            yield batch_x, batch_df["l1norm"], batch_df["seq_length"]

    def gen_epochs(self, num_epochs, batch_size):
        """
        Return an "epoch generator" over the input DataFrame.
        """
        for i in range(num_epochs):
            yield self.gen_batch(batch_size)
            self.shuffle()  # Suffle the input between epochs.


class GlorotIdentityInitializer(Initializer):
    """
    Initialize the recurrent weights with the identity matrix [0] and
    use the Glorot (Xavier) initialization for the non-recurrent weights.
    Initialize the bias weights to be zero.

    The non-recurrent weights multiply the input and
    the recurrent weights multiply the state.

    ..[0] Le, Jaitly, Hinton, "A Simple Way to Initialize Recurrent Networks
      of Rectified Linear Units", CoRR, vol abs/1504.00941, 2015.
    """

    def __init__(self, gain=1.0, dtype=tf.float32):
        self.dtype = dtype
        self._indentity_initializer = Identity(gain)
        self._glorot_uniform_initializer = glorot_uniform_initializer()

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        # `shape` is [state_size + num_classes, state_size]
        state_size = shape[1]
        num_classes = shape[0] - state_size

        # Non-recurrent weights
        w = self._glorot_uniform_initializer(
            [num_classes, state_size], dtype=dtype)
        # Recurrent weights
        u = self._indentity_initializer(
            [state_size, state_size], dtype=dtype, partition_info=partition_info)

        return tf.concat([w, u], axis=0)


class ReluRNNCell(BasicRNNCell):
    """
    Basic RNN cell with the ReLu activation.

    *  We have to subclass `BasicRNNCell` to be able to pass a kernel initializer.
    """

    def __init__(self, num_units, activation=tf.nn.relu, kernel_initializer=None, reuse=None):
        super(ReluRNNCell, self).__init__(num_units, activation=activation, reuse=reuse)
        self._kernel_initializer = kernel_initializer

    def call(self, inputs, state):
        """
        Basic RNN: output = new_state = act(W * input + U * state + B).

        The implentation is similar to `BasicRNNCell.call()`
        in tensorflow/python/ops/rnn_cell_impl.py, but we pass
        the kernel initializer to `_Linear()`
        """
        if self._linear is None:
            self._linear = _Linear([inputs, state], self._num_units, build_bias=True,
                                   kernel_initializer=self._kernel_initializer)

        output = self._activation(self._linear([inputs, state]))
        return output, output


class RNNModelConfig(object):
    """
    Configuration for the model.

    Attributes
    ----------
    learning_rate : float
        The learning rate value for the Adam algorithm
    max_grad_norm : float
        Maximum clipping value for the L2-norm of the gradients
    batch_size : int
        Size of the input batch
    state_size: int
        Size of the hidden state of the RNN cell
    keep_probability: float
        Dropout keep probability value (both for input and output)
    identity_init: bool
       If True use the "GlorotIdentity" initializer for the weights,
       else use the "Glorot" initializer
    """

    @classmethod
    def init_random(cls):
        """ Generate a random config. """
        learning_rate = 10 ** (-4 * np.random.random())  # log scale 0.0001 - 1
        if np.random.binomial(1, 0.8):
            max_grad_norm = np.random.choice([1, 10, 100, 1000])  # 1 - 1000
        else:
            max_grad_norm = None
        batch_size = 2 ** (np.random.randint(4, 9))  # 16 - 256
        state_size = np.random.choice([5, 10, 25, 50, 75, 100])  # 5 - 100
        identity_init = bool(np.random.binomial(1, 0.7))  # 70/30
        return cls(learning_rate=learning_rate, max_grad_norm=max_grad_norm,
                   batch_size=batch_size, state_size=state_size,
                   identity_init=identity_init)

    def __init__(self, learning_rate=0.01, max_grad_norm=100,
                 batch_size=64, state_size=10, keep_probability=1,
                 identity_init=True):
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.state_size = state_size
        self.keep_probability = keep_probability
        self.identity_init = identity_init

    def as_dict(self):
        return dict(
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            batch_size=self.batch_size,
            state_size=self.state_size,
            keep_probability=self.keep_probability,
            identity_init=self.identity_init
        )

    def __repr__(self):
        return "<RNNModelConfig: \n{}>".format(
            repr(pd.Series(data=self.as_dict())))


class RNNModel(object):
    """ Recurrent Neural Network with basic ReLu cells. """

    def __init__(self, config):

        # Parameters (see the docstring of RNNModelConfig)
        self.batch_size = batch_size = config.batch_size
        self.state_size = state_size = config.state_size
        self.learning_rate = learning_rate = config.learning_rate
        self.max_grad_norm = max_grad_norm = config.max_grad_norm
        self.keep_prob_value = config.keep_probability
        self.identity_init = identity_init = config.identity_init

        # Placeholders
        self.x = tf.placeholder(tf.int32, [batch_size, None])
        self.seq_length = tf.placeholder(tf.int32, [batch_size])
        self.y = tf.placeholder(tf.float32, [batch_size])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        # RNN inputs
        if identity_init:
            kernel_initializer = GlorotIdentityInitializer()
        else:
            kernel_initializer = glorot_uniform_initializer()

        cell = ReluRNNCell(state_size, kernel_initializer=kernel_initializer)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
            input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)

        # rnn_inputs is a tensor with shape [batch_size, ?, num_classes]
        rnn_inputs = tf.one_hot(self.x, NUM_CLASSES)
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell, rnn_inputs, sequence_length=self.seq_length, initial_state=init_state)

        # Get the last relevant output
        last_rnn_output = tf.gather_nd(rnn_outputs,
                                       tf.stack([tf.range(batch_size), self.seq_length - 1], axis=1))

        # Predictions, loss
        linear_w = tf.get_variable("linear_w", shape=[state_size, 1],
                                   initializer=tf.glorot_uniform_initializer())
        linear_b = tf.get_variable("linear_b", shape=[],
                                   initializer=tf.zeros_initializer())
        self.preds = tf.matmul(last_rnn_output, linear_w) + linear_b
        self.loss = tf.losses.mean_squared_error(
            tf.expand_dims(self.y, 1), self.preds)

        # Training step
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)

        if max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars), max_grad_norm)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

    def evaluate_batch(self, sess, x, y, seq_length):

        feed_dict = {self.x: x, self.y: y, self.seq_length: seq_length}
        preds, loss_value = sess.run([self.preds, self.loss],
                                     feed_dict=feed_dict)
        return loss_value, np.ravel(preds)

    def train_batch(self, sess, x, y, seq_length):

        feed_dict = {self.x: x, self.y: y,
                     self.seq_length: seq_length,
                     self.keep_prob: self.keep_prob_value}
        _, loss_value = sess.run([self.train_op, self.loss],
                                 feed_dict=feed_dict)
        return loss_value


def train_model(sess, model, train_input, num_epochs, verbose=True):
    """
    Train the model on the given input.

    Parameters
    ----------
    model : RNNModel
    train_input : RNNInput
    num_epochs : int
    verbose : bool

    Return
    ------
    list
        Training losses (averages per 100 batches)
    """
    train_losses = []

    for idx, epoch in enumerate(train_input.gen_epochs(num_epochs, model.batch_size)):
        train_loss = 0
        if verbose:
            print "EPOCH {}".format(idx)
        for step, (x, y, seq_length) in enumerate(epoch):
            train_loss_ = model.train_batch(sess, x, y, seq_length)
            if np.isnan(train_loss_):
                raise ValueError('Nan loss')
            train_loss += train_loss_
            if step % 100 == 0 and step > 0:
                if verbose:
                    print "Average loss at step {}: {}".format(step, train_loss / 100)
                train_losses.append(train_loss / 100)
                train_loss = 0

    return train_losses


def evaluate_model(sess, model, input_):
    """
    Evaluate the model on the given input.

    Parameters
    ----------
    model : RNNModel
    input_ : RNNInput

    Return
    ------
    float
        Prediction loss
    list
        Predictions "zipped" together with the ground truths
    """
    pred_losses = []
    preds = []

    for x, y, seq_length in input_.gen_batch(model.batch_size):
        pred_loss_, preds_ = model.evaluate_batch(sess, x, y, seq_length)
        pred_losses.append(pred_loss_)
        preds.append(zip(preds_, y))

    preds = list(itertools.chain(*preds))

    pred_loss = np.mean(pred_losses)
    return pred_loss, preds
