import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




class Data():
    def __init__(self):

        # generate data set
        t = np.arange(0, 60, 1)
        y_predict = np.sin(0.5 * t)
        y_factor1 = np.sin(0.5 * t - 5)
        y_factor2 = np.sin(0.5 * t - 3)

        # y_predict must be listed first, index 0 in the numpy array is the predicted variable
        dataframe = pd.DataFrame(index=t, data={'y_predict': y_predict, 'y_factor1': y_factor1,
                                                'y_factor2': y_factor2})

        # try with only one variable
        dataframe = pd.DataFrame(index=t, data={'y_predict': y_predict})

        self.train = dataframe[0:50]
        self.test = dataframe[50:]


    def next_batch(self, batch_size, input_vec_len, time_steps_shifted, train_data):
        # pull a bunch of random samples
        rand_start = np.random.randint(0, len(train_data) - (input_vec_len+time_steps_shifted), size=batch_size)

        indexes = np.repeat(np.array(range(input_vec_len+time_steps_shifted)).reshape(1, -1), axis=0, repeats=batch_size)

        # add the start values to the initialized index vectors
        indexes += rand_start.reshape(-1, 1)

        train_data_array = np.array(train_data)

        values = np.take(train_data_array, indices=indexes, axis=0)

        x = values[:, 0:-time_steps_shifted, :]

        # index 0 is the variable we are trying to predict
        y = values[:, time_steps_shifted:, 0]

        y = y.reshape(np.shape(y)[0], np.shape(y)[1], 1)

        return x, y





n_steps = 5
n_inputs = 1
n_neurons = 100
n_outputs = 1
time_shift = 2

data_obj = Data()
x_batch, y_batch = data_obj.next_batch(batch_size=15, input_vec_len=5, time_steps_shifted=2, train_data=data_obj.train)


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, activation=tf.nn.tanh)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])



learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_iterations = 60000
batch_size = 10

with tf.Session() as sess:

    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = data_obj.next_batch(batch_size=15, input_vec_len=n_steps,
                                               time_steps_shifted=time_shift, train_data=data_obj.train)

        # arb_axis = np.arange(0, n_steps+time_shift, 1)
        # plt.plot(arb_axis[0:n_steps], X_batch[0, :, 0], color='red', alpha=0.5)
        # plt.plot(arb_axis[time_shift:], y_batch[0, :, 0], color='blue', alpha=0.5)
        # plt.show()

        sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batch, y: y_batch})
            print(iteration, '\tMSE:', mse)












