import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

input_size = 784
output_size = 10
hidden_layer1_size = 4000
hidden_layer2_size = 4000
hidden_layer3_size = 4000
hidden_layer4_size = 4000
hidden_layer5_size = 4000
hidden_layer6_size = 4000
hidden_layer7_size = 4000
hidden_layer8_size = 4000
hidden_layer9_size = 4000
hidden_layer10_size = 4000

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer1_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer1_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("wights_2", [hidden_layer1_size, hidden_layer2_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer2_size])
outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer2_size, hidden_layer3_size])
biases_3 = tf.get_variable("biases_3", [hidden_layer3_size])
outputs_3 = tf.nn.relu(tf.matmul(outputs_2, weights_3) + biases_3)

weights_4 = tf.get_variable("weights_4", [hidden_layer3_size, hidden_layer4_size])
biases_4 = tf.get_variable("biases_4", [hidden_layer4_size])
outputs_4 = tf.nn.relu(tf.matmul(outputs_3, weights_4) + biases_4)

weights_5 = tf.get_variable("weights_5", [hidden_layer4_size, hidden_layer5_size])
biases_5 = tf.get_variable("biases_5", [hidden_layer5_size])
outputs_5 = tf.nn.relu(tf.matmul(outputs_4, weights_5) + biases_5)

weights_6 = tf.get_variable("weights_6", [hidden_layer5_size, hidden_layer6_size])
biases_6 = tf.get_variable("biases_6", [hidden_layer6_size])
outputs_6 = tf.nn.relu(tf.matmul(outputs_5, weights_6) + biases_6)

weights_7 = tf.get_variable("weights_7", [hidden_layer6_size, hidden_layer7_size])
biases_7 = tf.get_variable("biases_7", [hidden_layer7_size])
outputs_7 = tf.nn.relu(tf.matmul(outputs_6, weights_7) + biases_7)

weights_8 = tf.get_variable("weights_8", [hidden_layer7_size, hidden_layer8_size])
biases_8 = tf.get_variable("biases_8", [hidden_layer8_size])
outputs_8 = tf.nn.relu(tf.matmul(outputs_7, weights_8) + biases_8)

weights_9 = tf.get_variable("weights_9", [hidden_layer8_size, hidden_layer9_size])
biases_9 = tf.get_variable("biases_9", [hidden_layer9_size])
outputs_9 = tf.nn.relu(tf.matmul(outputs_8, weights_9) + biases_9)

weights_10 = tf.get_variable("weights_10", [hidden_layer9_size, hidden_layer10_size])
biases_10 = tf.get_variable("biases_10", [hidden_layer10_size])
outputs_10 = tf.nn.relu(tf.matmul(outputs_9, weights_10) + biases_10)

weights_final = tf.get_variable("weights_final", [hidden_layer10_size, output_size])
biases_final = tf.get_variable("biases_final", [output_size])
outputs = tf.matmul(outputs_10, weights_final) + biases_final

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)
optimize = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(mean_loss)

out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()
sess.run(initializer)

batch_size = 150
batches_number = mnist.train._num_examples // batch_size
max_epochs = 50
prev_validation_loss = 9999999.

for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0.
    for batch_counter in range(batches_number):
        input_batch, target_batch = mnist.train.next_batch(batch_size)
        _, batch_loss = sess.run([optimize, mean_loss], feed_dict={inputs: input_batch, targets: target_batch})
        curr_epoch_loss += batch_loss

    curr_epoch_loss /= batches_number

    validation_input_batch, validation_target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                                        feed_dict={inputs: validation_input_batch, targets: validation_target_batch})

    print('Epoch '+str(epoch_counter+1) +
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss) +
          '. Validation loss: '+'{0:.3f}'.format(validation_loss) +
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.0) + '%')

    if validation_loss > prev_validation_loss:
        break
    prev_validation_loss = validation_loss
print('end of training.')

test_input_batch, test_target_batch = mnist.test.next_batch(mnist.test._num_examples)
test_accuracy = sess.run([accuracy], feed_dict={inputs: test_input_batch, targets: test_target_batch})
test_accuracy_percent = test_accuracy[0] * 100
print('Test accuracy: ' + '{0: .2f}'.format(test_accuracy_percent)+'%')
