'''
A linear regression learning algorithm using TensorFlow library for two variables.
Author: Kevin Toms.
Project: https://github.com/kevintomsgithub/Linear-Regression-Raw-Tensorflow-Two-Variables
'''

# Importing Libraries.
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# Ruling Parameters.
learning_rate = 0.001 # -- parameter for optimizer.
training_epoch = 1000 # -- parameter for number of training steps.
display_step = 100    # -- parameter for logging at each 100 steps.

# Dataset for training
train_data_X1 = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
train_data_X2 = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
train_data_Y = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0,12.0, 14.0, 16.0, 18.0, 20.0])

# Dataset for testing.
test_data_X1 = np.asarray([12.0, 13.0, 14.0, 15.0, 16.0])
test_data_X2 = np.asarray([12.0, 13.0, 14.0, 15.0, 16.0])
test_data_Y = np.asarray([24.0, 26.0, 28.0, 30.0, 32.0])

# Value to be predicted after training, where x = 30 and y =31.
predict_X1 = 30
predict_X2 = 30

# Tensorflow variables through which input data (train_data and test_data) flow.
X1 = tf.placeholder(dtype='float32')
X2 = tf.placeholder(dtype='float32')
Y = tf.placeholder(dtype='float32')

# Tensorflow variables that hold values for 'weight' and 'bias' as in the equation y = W * X + b.
W1 = tf.Variable(0.0, name='weights')
W2 = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Linear Regression Function same as, y = m*x + c (linear algebra).
model = (W1 * X1) + (W2 * X2) + b
# More technically we can write the above function as,
# model = tf.add(tf.add(tf.multiply(W1,X1), tf.multiply(W2,X2)), b)

# Error - Mean Squared error, gives the error during training.
cost = tf.reduce_sum(tf.pow(model-Y, 2))

# Optimizer - Plays the vital role in training, the optimizer magically flows the input data,
# to the model variable by minimizing the cost(error).
# Different optimizers are used, for different train dataset.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing all the declared tensorflow variables.
init = tf.global_variables_initializer()

# Session starts for performing operations with tensorflow variables.
with tf.Session() as sess:

	# Variables are being initialized.
	sess.run(init)

	# Training starts... training_epoch = number of times the training steps have to be repeated.
	for epoch in range(training_epoch):

		# For each training step the model is fed with the whole input training data.
		for (x1, x2, y) in zip(train_data_X1, train_data_X2, train_data_Y):
			# Optimizer takes in the training data via, feed dictionary( feed_dict ),
			# Single train data is being passed in an iteration.
			sess.run(optimizer, feed_dict={X1: x1, X2: x2, Y: y})

		# Display Logs per 100 steps as defined earlier.
		if (epoch+1) % display_step == 0:
			loss = sess.run(cost, feed_dict={X1: test_data_X1, X2: test_data_X2, Y: test_data_Y})
			print('At Epoch :', epoch+1,
					  'loss =', loss,
					  'W1 =', sess.run(W1),
					  'W2 =', sess.run(W2),
						'b =', sess.run(b))

	print('Training Finished...')

	# Calculate the final training cost.
	train_cost = sess.run(cost, feed_dict={X1: train_data_X1, X2: train_data_X2, Y: train_data_Y})
	print('Final Loss :', train_cost)
	print('Final Weights W1 : ', sess.run(W1), ' W2 : ', sess.run(W2), 'Final Bias :', sess.run(b))

	# Graphically Display - dataset of training and the new model obtained.
	plt.plot(train_data_X1, train_data_X2, train_data_Y, 'ro', label='Train data')
	plt.plot(train_data_X1, train_data_X2, (sess.run(W1) * train_data_X1) + (sess.run(W2) * train_data_X2) + sess.run(b), label='Fitted line')
	plt.legend()
	plt.show()

	# Testing the model
	print('Testing...')
	# Estimating the final loss from a set of test dataset.
	test_cost = sess.run(cost, feed_dict={X1: test_data_X1, X2: test_data_X2, Y: test_data_Y})
	print('Testing Finished...')
	print('Testing cost :', test_cost)
	print('Absolute mean Square loss difference :',abs(train_cost - test_cost))

	# Graphically Display - dataset of testing and the new model obtained.
	plt.plot(test_data_X1, test_data_X2, test_data_Y, 'bo', label='Test data')
	plt.plot(test_data_X1, test_data_X2, (sess.run(W1) * test_data_X1) + (sess.run(W2) * test_data_X2) + sess.run(b), label='Fitted Line')
	plt.legend()
	plt.show()

	# Predict the value of y for x = 30.
	print('Prediction...')
	predict_Y = (sess.run(W1) * predict_X1) + (sess.run(W2) * predict_X2) + sess.run(b)
	print('Prediction for x1 = ',predict_X1, ' x2 : ', predict_X2, ' y = ',predict_Y)