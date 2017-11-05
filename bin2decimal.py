import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

# Convert the binary no to decimal
#	bin = [0,1,1]	
#	Return: [0,1,1] => 3
def bin2int(bin_list):
	bin_str = ""
	for k in bin_list:
		bin_str += str(int(k))
	return int(bin_str, 2)
	
# Generate the dataset, i.e., the binary number and their answers
# 	num = 1000	: size of the dataset
#	bin_len = 8	: the size of the binary number to be genrated
#	Returns: it returns the x and y. The x contains the binary number and y contains their answers 	
def dataset(num, bin_len):
	x = np.zeros((num, bin_len))
	y = np.zeros((num))

	for i in range(num):
		x[i] = np.round(np.random.rand(bin_len)).astype(int)
		y[i] = bin2int(x[i])
	return x, y	

import tensorflow as tf
from tensorflow.contrib import rnn	

# Parameters
x_len = 8
no_of_samples = 1000
lr = 0.01
training_steps= 50000
display_step = 5000

n_input = x_len
timestep = 1
n_hidden = 16
n_output = 1

# Training and tsting dataset
trainX, trainY = dataset(no_of_samples, x_len)
testX, testY = dataset(20, x_len)

# Graph input 
X = tf.placeholder(tf.float32, [None, timestep, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

# Weights and bias
weights = tf.Variable(tf.random_normal([n_hidden, n_output]))
bias = tf.Variable(tf.random_normal([n_output]))

# The RNN model
def RNN(x, W, b):
	x = tf.unstack(x, timestep, 1)
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], W) + b

logits = RNN(X, weights, bias)

trainX = np.reshape(trainX, [-1, timestep, n_input])
trainY = np.reshape(trainY, [-1, n_output])

testX = np.reshape(testX, [-1, timestep, n_input])
testY = np.reshape(testY, [-1, n_output])

loss = tf.reduce_mean(tf.losses.mean_squared_error(logits, Y))
optimizer = tf.train.RMSPropOptimizer(lr)
train = optimizer.minimize(loss)

with tf.Session() as sess:
	tf.global_variables_initializer().run()

	for step in range(training_steps):
		_, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
		if (step+1) % display_step == 0:
			print "Step: ", step+1, "\tLoss: ", _loss

	print("Optimization Finished")

	result = sess.run(logits, feed_dict={X: testX})
	result = sess.run(tf.round(result))

	print "Real \t Guess"
	for i in range(20):
		if testY[i] == result[i]:
			print "True"
		else:
			print "False"	
			print testY[i], ' -> ', result[i] 


