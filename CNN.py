import os
import cv2
import numpy as np
import tensorflow as tf
from MNIST import load_data
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)
CUR_DIR = os.getcwd()
MODEL_SAVE_PATH = os.path.join(CUR_DIR, "model")
if not os.path.exists(MODEL_SAVE_PATH):
	os.mkdir(MODEL_SAVE_PATH)
MODEL_NAME = "CNN.ckpt"
LOG_DIR = "TensorBoard/"
TRAINING_STEPS = 10000
BATCH_SIZE = 100
LEARNING_RATE = 0.05
DROPOUT_PROB = 0.7
SAVE_PATH = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
Training = True

class CNN(object):

	def __init__(self, num_classes= 10, isTraining= Training, dropout_keep_prob= DROPOUT_PROB, prediction_fn= slim.softmax, scope= "LetNet"):
		self.isTraining = isTraining
		self.num_classes = num_classes
		if self.isTraining:
			self.y = tf.placeholder(tf.float32, [None, self.num_classes], name= "Label")
			pass
		self.x = tf.placeholder(tf.float32, [None, 1024], name= "Inputs")
		self.images = tf.reshape(self.x, [-1, 32, 32, 1], name= "Reshape")
		self.dropout_keep_prob = dropout_keep_prob
		self.prediction_fn = prediction_fn
		self.scope = scope
		self.END_POINT = {}
		predict = self.Net()
		self.Predict = predict
		pass

	def Net(self):
		predict = {}
		with tf.variable_scope(name_or_scope= self.scope, default_name= "LetNet", values= [self.images]):
			#net = slim.repeat(self.images , 0, slim.conv2d, num_outputs= 6, kernel_size= [5, 5], stride= 1, padding= "same", activation_fn= tf.nn.relu, scope= "Conv_1")
			net = slim.repeat(self.images , 1, slim.conv2d, num_outputs= 6, kernel_size= [5, 5], stride= 1, padding= "valid", activation_fn= tf.nn.relu, scope= "Conv_1")
			self.END_POINT['conv1'] = net
			net = slim.max_pool2d(net, [2, 2], stride= 2, padding= "valid", scope= "Pool_1")
			self.END_POINT['pool1'] = net
			#net = slim.repeat(net , 0, slim.conv2d, num_outputs= 6, kernel_size= [5, 5], stride= 1, padding= "same", activation_fn= tf.nn.relu, scope= "Conv_3")
			net = slim.repeat(net , 1, slim.conv2d, num_outputs= 16, kernel_size= [5, 5], stride= 1, padding= "valid", activation_fn= tf.nn.relu, scope= "Conv_2")
			self.END_POINT['conv2'] = net
			net = slim.max_pool2d(net, [2, 2], stride= 2, padding= "valid", scope= "Pool_2")
			self.END_POINT['pool2'] = net
			#net = slim.repeat(net , 0, slim.conv2d, num_outputs= 16, kernel_size= [5, 5], stride= 1, padding= "same", activation_fn= tf.nn.relu, scope= "Conv_5")
			net = slim.repeat(net , 1, slim.conv2d, num_outputs= 120, kernel_size= [5, 5], stride= 1, padding= "valid", activation_fn= tf.nn.relu, scope= "Conv_3")
			self.END_POINT['conv3'] = net
			net = slim.fully_connected(net, num_outputs= 84, activation_fn= tf.nn.relu, scope= "FC_1")
			self.END_POINT['fc1'] = net
			
			if not self.num_classes:
				return net, self.END_POINT

			net = slim.dropout(net, self.dropout_keep_prob, is_training= self.isTraining, scope= "Dropout_1")
			self.END_POINT['Dropout'] = net
			logits = slim.fully_connected(net, num_outputs= self.num_classes, activation_fn= None, scope= "FC_2")
			self.END_POINT['Logits'] = net
			flat = slim.flatten(logits, scope= "Flatten")

		cls = tf.argmax(flat, 1, name= "Class_Arg_Max")
		prob = self.prediction_fn(flat, scope= "Probability")
		
		if self.isTraining:
			loss = tf.reduce_mean( - tf.reduce_sum(self.y * tf.log(prob + 1e-9), reduction_indices= 1), name= "Cross_Entropy")
			predict["Loss"] = loss
			tf.summary.scalar("Loss", loss)
			
			with tf.name_scope("Gradient_Descent"):
				self.optimizer = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(loss)
			with tf.name_scope("Accuracy"):
				correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1), name= "Correct_Predict")
				acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
				predict["Accuracy"] = acc
				tf.summary.scalar("Accuracy", acc)
				pass
			pass
		predict["Class"] = cls[0]
		predict["Prob"] = prob
		self.END_POINT['Predict'] = predict

		return predict


if __name__ == "__main__":
	#x_train, y_train, x_test, y_test = load_data()
	
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	x_train = mnist.train.images # [5500, 784]
	y_train = mnist.train.labels # [5500,  10]
	x_test = mnist.test.images   # [500,  784]
	y_test = mnist.test.labels   # [500,   10]
		
	cfg = tf.ConfigProto(allow_soft_placement= True)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.8)
	cfg.gpu_options.allow_growth = True
	sess = tf.Session(config= cfg)
	
	Net = CNN(isTraining= Training)
	
	pred = Net.Predict

	SAVER = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(LOG_DIR, graph = sess.graph)

	if not Training:
		models = os.listdir(MODEL_SAVE_PATH)
		for file in models:
			if '.ckpt' in file:
				SAVER.restore(sess, SAVE_PATH)
				break
			pass
		x = Net.x
		imgs = x_test[:]
		for i,img in enumerate(imgs):
			img = cv2.resize(img, (28, 28))
			img = cv2.resize(img, (32, 32), interpolation= cv2.INTER_LANCZOS4)
			img = np.reshape(img, (1, img.size))
			ans = np.argmax(y_test[i])
			predict = sess.run(pred, feed_dict= {x: img})
			print (f"Classes : {predict['Class']} \nProb : {predict['Prob'][0][predict['Class']]} \nCorrect : {ans} \n")
	else:
		x = Net.x
		y = Net.y
		op = Net.optimizer
		for step in range(TRAINING_STEPS):
			x_train_batch, y_train_batch = mnist.train.next_batch(BATCH_SIZE) # x [batch_size, 784] , y [batch_size, 10]
			
			x_batch = []
			for img in x_train_batch:
				img = cv2.resize(img, (32, 32), interpolation= cv2.INTER_LANCZOS4)
				img = np.reshape(img, (img.size,))
				x_batch.append(img)
			
			sess.run(op, feed_dict= {x: x_batch, y:y_train_batch})
			if step % BATCH_SIZE == 0:
				predict = sess.run(pred, feed_dict= {x: x_batch, y:y_train_batch})
				summary = sess.run(merged, feed_dict= {x: x_batch, y:y_train_batch})
				writer.add_summary(summary, step)
				ans = np.argmax(y_train_batch)
				print (f"Classes : {predict['Class']} \nProb : {predict['Prob'][0][predict['Class']]} \nCorrect : {ans} \n")
				print (f"Step : {step} \nLoss : {predict['Loss']} \nAccuracy : {predict['Accuracy']} \n")
				pass
			pass
		SAVER.save(sess, SAVE_PATH)
	sess.close()