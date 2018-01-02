import tensorflow as tf
import numpy as np

class DataReader():
    def __init__(self, dataset, batch_size = None):
        npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size

    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()

        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1

        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1

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

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, hidden_layer_size])
biases_3 = tf.get_variable("biases_3", [hidden_layer_size])
outputs_3 = tf.nn.sigmoid(tf.matmul(outputs_2, weights_3) + biases_3)

weights_final = tf.get_variable("weights_final", [hidden_layer_size, output_size])
biases_final = tf.get_variable("biases_final", [output_size])
outputs = tf.matmul(outputs_3, weights_final) + biases_final

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)

out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

optimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mean_loss)

sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()
sess.run(initializer)

batch_size = 100
max_epoch = 500
prev_validation_loss = 99999999

train_data = DataReader('train', batch_size)
validation_data = DataReader('validation')

for epoch_counter in range(max_epoch):
    curr_epoch_loss = 0.

    for batch_train_inputs, batch_train_targets in train_data:
        _, batch_loss = sess.run([optimize, mean_loss],
                                 feed_dict={inputs: batch_train_inputs, targets: batch_train_targets})
        curr_epoch_loss += batch_loss

    curr_epoch_loss /= train_data.batch_count

    validation_loss = 0.
    validation_accuracy = 0.
    for batch_validation_inputs, batch_validation_targets in validation_data:
        validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                                                        feed_dict={inputs: batch_validation_inputs,
                                                                   targets: batch_validation_targets})

    print('Epoch '+str(epoch_counter+1) +
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss) +
          '. Validation loss: '+'{0:.3f}'.format(validation_loss) +
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')

    if validation_loss > prev_validation_loss:
        break
    prev_validation_loss = validation_loss

print('End of training.')

test_data = DataReader('test')
for inputs_batch, targets_batch in test_data:
    test_accuracy = sess.run([accuracy], feed_dict={inputs: inputs_batch, targets: targets_batch})

test_accuracy_percent = test_accuracy[0] * 100.
print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')