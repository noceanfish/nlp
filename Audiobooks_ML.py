import tensorflow as tf
import numpy as np

class DataReader():
    def __init__(self, dataset, batch_size = None):
        npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))
        self.input, self.targets = np['inputs'].astype(np.float), np['targets'].astype(np.int)

        if batch_size is None:
            self.batch_size = self.input.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.input.shape[0] // self.batch_size

    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()

        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.input[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1

        classes_num = 2
        targets_one_hot = np.zeros(targets_batch.shape[0], classes_num)
        targets_one_hot[range(targets_batch.shpe[0]), targets_batch] = 1

        return inputs_batch, targets_one_hot

    def __iter__(self):
        return self

input_size = 10
output_size = 2
hidden_layer_size = 50

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])
outputs_2 = tf.nn.sigmoid(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [outputs_2, hidden_layer_size])
biases_3 = tf.get_variable("biases_3", [hidden_layer_size])
outputs_3 = tf.nn.
weights_final =relu(tf.matmul(outputs_2, weights_3) + biases_3)




