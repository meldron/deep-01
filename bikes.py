# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
import sys

from sklearn.metrics import mean_squared_error

data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

test_data = data[-21 * 24:]
data = data[:-21 * 24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        self.activation_function_prime = lambda x: x * (1 - x)

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###

        output_errors = targets - final_outputs  # Output layer error is the difference between desired target and actual output.

        # TODO: Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)  # errors propagated to the hidden layer
        hidden_grad = self.activation_function_prime(hidden_outputs)  # hidden layer gradients

        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr * output_errors * hidden_outputs.T  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * np.dot(hidden_errors * hidden_grad, inputs.T)

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs

        return final_outputs


# In[11]:

def MSE(y, Y):
    return np.mean((y - Y) ** 2)


epochs = 100
learning_rates = [0.25, 0.2, 0.15, 0.1, 0.07]
hidden_nodes = [3, 5, 7, 9]
output_nodes = [1, 2, 3, 4]

N_i = train_features.shape[1]
mean_squared_errors = []

np.random.seed(19)

for lr in learning_rates:
    for hn in hidden_nodes:
        for on in output_nodes:
            network = NeuralNetwork(N_i, hn, on, lr)

            losses = {'train': [], 'validation': []}
            for e in range(epochs):
                # Go through a random batch of 128 records from the training data set
                batch = np.random.choice(train_features.index, size=128)
                for record, target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt']):
                    network.train(record, target)
                # Printing out the training progress
                train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
                val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
                progress = 100 * e / float(epochs)

                o = u"\rLR: {:0.2f} ... HL: {:d} ... OL: {:d} ... Progress: {:04.1f}% ... " \
                    u"Training loss: {:0.5f} ... Validation loss: {:0.5f}"\
                    .format(lr, hn, on, progress, train_loss, val_loss)

                sys.stdout.write(o)
                losses['train'].append(train_loss)
                losses['validation'].append(val_loss)
            mean_squared_errors.append((lr, hn, on, train_loss, val_loss))
        sys.stdout.write("\n")
df = pd.DataFrame(mean_squared_errors)
df.columns = ['LearningRate', 'HiddenLayer', 'OutputLayer', "TrainLoss", "ValidationLoss"]
df.to_csv('evaluation.csv', index=False)
print(df)

"""
fig, ax = plt.subplots(figsize=(8, 4))

mean, std = scaled_features['cnt']
predictions_raw = network.run(test_features)
predictions = network.run(test_features)*std + mean

real_test_data = (test_targets['cnt'] * std + mean).values
error_data = real_test_data - predictions[0]

ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.plot(predictions[0], label='Prediction')
ax.plot(error_data, label='Error')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
"""

# inputs = [0.5, -0.2, 0.1]
# targets = [0.4]
# test_w_i_h = np.array([[0.1, 0.4, -0.3],
#                        [-0.2, 0.5, 0.2]])
# test_w_h_o = np.array([[0.3, -0.1]])
#
#
# class TestMethods(unittest.TestCase):
#     ##########
#     # Unit tests for data loading
#     ##########
#
#     def test_data_path(self):
#         # Test that file path to dataset has been unaltered
#         self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
#
#     def test_data_loaded(self):
#         # Test that data frame loaded
#         self.assertTrue(isinstance(rides, pd.DataFrame))
#
#     ##########
#     # Unit tests for network functionality
#     ##########
#
#     def test_activation(self):
#         network = NeuralNetwork(3, 2, 1, 0.5)
#         # Test that the activation function is a sigmoid
#         self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))
#
#     def test_train(self):
#         # Test that weights are updated correctly on training
#         network = NeuralNetwork(3, 2, 1, 0.5)
#         network.weights_input_to_hidden = test_w_i_h.copy()
#         network.weights_hidden_to_output = test_w_h_o.copy()
#
#         network.train(inputs, targets)
#         self.assertTrue(np.allclose(network.weights_hidden_to_output,
#                                     np.array([[0.37275328, -0.03172939]])))
#         self.assertTrue(np.allclose(network.weights_input_to_hidden,
#                                     np.array([[0.10562014, 0.39775194, -0.29887597],
#                                               [-0.20185996, 0.50074398, 0.19962801]])))
#
#     def test_run(self):
#         # Test correctness of run method
#         network = NeuralNetwork(3, 2, 1, 0.5)
#         network.weights_input_to_hidden = test_w_i_h.copy()
#         network.weights_hidden_to_output = test_w_h_o.copy()
#
#         self.assertTrue(np.allclose(network.run(inputs), 0.09998924))
#
# # suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
# # unittest.TextTestRunner().run(suite)
