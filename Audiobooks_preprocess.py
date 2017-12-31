import numpy as np
from sklearn import preprocessing

Audiobooks = np.loadtxt('Audiobooks-data.csv', delimiter=',')
raw_input = Audiobooks[:, 1:-1]
raw_target = Audiobooks[:, -1]

# balance dataset
target_counter = int(np.sum(raw_target[:]))
num = 0
indeice_remove = []
for i in range(raw_target.shape[0]):
    if raw_target[i] == 0:
        num += 1
        if num >= target_counter:
            indeice_remove.append(i)

balance_input = np.delete(raw_input, indeice_remove, axis=0)
balance_target = np.delete(raw_target, indeice_remove, axis=0)

# scale input
scaled_inputs = preprocessing.scale(balance_input)

# shuffled data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = balance_target[shuffled_indices]

# split data into train validation test
samples_count = shuffled_inputs.shape[0]

train_count = int(samples_count * 0.8)
validation_count = int(samples_count * 0.1)
test_count = samples_count - train_count - validation_count

train_input = shuffled_inputs[:train_count]
train_target = shuffled_targets[:train_count]

validation_input = shuffled_inputs[train_count:(train_count+validation_count)]
validation_target = shuffled_targets[train_count:(train_count+validation_count)]

test_input = shuffled_inputs[train_count+validation_count:]
test_target = shuffled_targets[train_count+validation_count:]

print(np.sum(train_target), train_count, np.sum(train_target) / train_count)
print(np.sum(validation_target), validation_count, np.sum(validation_target) / validation_count)
print(np.sum(test_target), test_count, np.sum(test_target) / test_count)

np.savez('Audiobooks_data_train', inputs=train_input, targets=train_target)
np.savez('Audiobooks_data_validation', inputs=validation_input, targets=validation_target)
np.savez('Audiobooks_data_test', inputs=test_input, targets=test_target)