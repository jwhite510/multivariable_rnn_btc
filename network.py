import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linearize_and_process import get_linear_data

def normalize_values(values, scaler=None):

    if scaler:
        max_vals = scaler['max_vals']
        offset = scaler['offset']
        pass

    else:
        max_vals = np.expand_dims(np.max(values, axis=1), axis=1)
        offset = np.expand_dims(np.min(values, axis=1), axis=1)

    numerator = (values - offset)
    denom = (max_vals - offset)

    # correct for params that dont change in time by setting them to 1/2
    constant_value_indexes = np.repeat(denom == 0, np.shape(numerator)[1], axis=1)
    numerator[constant_value_indexes] = 1
    denom[denom == 0] = 2
    new_values = numerator / denom

    scale_mat = {}
    scale_mat['offset'] = offset
    scale_mat['max_vals'] = max_vals

    return new_values, scale_mat


def rescale(values, scalemat):

    # initialize output vector
    output_values = np.ones_like(values)

    # add the constant values
    indexes_2d = scalemat['offset'] == scalemat['max_vals']
    sample_time_length =  np.shape(values)[1]
    indexes_3d = np.repeat(indexes_2d, sample_time_length, axis=1)
    output_values[indexes_3d] = np.repeat(scalemat['max_vals'], sample_time_length, axis=1)[indexes_3d]

    # add the changing values
    indexes_2d_not_constant = scalemat['offset'] != scalemat['max_vals']
    indexes_3d_not_constant = np.repeat(indexes_2d_not_constant, sample_time_length, axis=1)
    # multiply by the magnitude
    magnitude =  np.repeat((scalemat['max_vals'] - scalemat['offset']), sample_time_length, axis=1)
    output_values[indexes_3d_not_constant] = values[indexes_3d_not_constant] * magnitude[indexes_3d_not_constant]
    # add the offset
    offset = np.repeat((scalemat['offset']), sample_time_length, axis=1)
    output_values[indexes_3d_not_constant] = output_values[indexes_3d_not_constant] +  offset[indexes_3d_not_constant]

    return output_values



class Data():
    def __init__(self):

        # retrieve data
        globaltime, pricedata = get_linear_data(dt=60)

        # generate data set
        # t = np.arange(0, 60, 1)
        # y_predict = np.sin(0.5 * t)
        # y_factor1 = np.sin(0.5 * t - 5)
        # y_factor2 = np.sin(0.5 * t - 3)

        # y_predict must be listed first, index 0 in the numpy array is the predicted variable
        dataframe = pd.DataFrame(index=globaltime, data={'y_predict': pricedata['btcprices'],
                                                         'y_factor1': pricedata['tetherprices'],
                                                         'y_factor2': pricedata['ethereumprices'],
                                                         'y_factor3': pricedata['bchprices']
                                                         })

        # try with only one variable
        # dataframe = pd.DataFrame(index=t, data={'y_predict': y_predict})
        timemax = globaltime[-1]
        timemin = globaltime[0]
        duration = timemax - timemin
        train_test_split = 0.8
        index_mid = timemin + train_test_split * duration

        self.train = dataframe[timemin:index_mid]
        self.test = dataframe[index_mid:timemax]


    def next_batch(self, batch_size, input_vec_len, time_steps_shifted, train_data):

        ########################
        ########################
        ########################
        ########################
        ########################
        # dont forget to get rid of this!!!!!!!!!!!!!!!!
        np.random.seed(998)
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################


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

        # normalize the values
        input_x_axis = np.array(range(input_vec_len))
        output_x_axis = input_x_axis + time_steps_shifted

        fig = plt.figure()
        gs = fig.add_gridspec(3,2)
        ax = fig.add_subplot(gs[0,:])
        ax.plot(input_x_axis, x[0, :, 0])
        # ax.plot(input_x_axis, x[0, :, 1])
        # ax.plot(input_x_axis, x[0, :, 2])
        # ax.plot(input_x_axis, x[0, :, 3])
        ax.plot(output_x_axis, y[0, :, 0], linestyle='dashed', color='blue')


        # normalize the input values between 0 and 1
        x_new, x_scale_mat = normalize_values(x)

        # take the scale values from the input to use on the output for training
        offset_y = np.expand_dims(x_scale_mat['offset'][:,:,0], axis=1)
        max_vals_y = np.expand_dims(x_scale_mat['max_vals'][:,:,0], axis=1)
        y_norm_mat = {}
        y_norm_mat['offset'] = offset_y
        y_norm_mat['max_vals'] = max_vals_y

        # scale the output with the same factors as the input
        y_new, y_scale_mat = normalize_values(y, scaler=y_norm_mat)


        ax = fig.add_subplot(gs[1, :])
        ax.plot(input_x_axis, x_new[0, :, 0])
        # ax.plot(input_x_axis, x_new[0, :, 1])
        # ax.plot(input_x_axis, x_new[0, :, 2])
        # ax.plot(input_x_axis, x_new[0, :, 3])
        ax.plot(output_x_axis, y_new[0, :, 0], linestyle='dashed', color='blue')

        x_rescaled = rescale(values=x_new, scalemat=x_scale_mat)
        y_rescaled = rescale(values=y_new, scalemat=y_norm_mat)

        ax = fig.add_subplot(gs[2, :])
        ax.plot(input_x_axis, x_rescaled[0, :, 0])
        # ax.plot(input_x_axis, x_rescaled[0, :, 1])
        # ax.plot(input_x_axis, x_rescaled[0, :, 2])
        # ax.plot(input_x_axis, x_rescaled[0, :, 3])
        ax.plot(output_x_axis, y_rescaled[0, :, 0], linestyle='dashed', color='blue')








        plt.ioff()
        plt.show()
        exit(0)


        return x, y





n_steps = 50
n_inputs = 4
n_neurons = 100
n_outputs = 1
time_shift = 20

data_obj = Data()

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
plt.ion()
with tf.Session() as sess:

    init.run()
    for iteration in range(n_iterations):
        x_batch, y_batch = data_obj.next_batch(batch_size=3, input_vec_len=n_steps,
                                               time_steps_shifted=time_shift, train_data=data_obj.train)



        sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        if iteration % 10 == 0:

            mse = loss.eval(feed_dict={X: x_batch, y: y_batch})
            print(iteration, '\tMSE:', mse)
            projections = sess.run(outputs, feed_dict={X: x_batch})


            arb_axis = np.arange(0, n_steps + time_shift, 1)
            plt.clf()
            # plot the inputs
            for i in range(n_inputs):
                if i == 0:
                    plt.plot(arb_axis[0:n_steps], x_batch[0, :, i], color='red', alpha=0.5)
                else:
                    plt.plot(arb_axis[0:n_steps], x_batch[0, :, i], color='black', alpha=0.5)

            plt.plot(arb_axis[time_shift:], y_batch[0, :, 0], color='blue', alpha=0.5, label='true value')
            plt.plot(arb_axis[time_shift:], projections[0, :, 0], color='orange', label='projected value')
            plt.legend(loc=2)
            plt.pause(0.1)














