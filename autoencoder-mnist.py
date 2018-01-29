'''
increase epochs.
Lower lr after epoch 20.
'''
# epochs completed = 21

import tensorflow as tf
import numpy as np
# from cv2 import imread, imshow
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("E:/workspace_py/mnist/", one_hot=True)
# print(mnist)
# base_path = "E:/workspace_py/datasets/catsdogs/train/train/"
# data = []

# for i in range(5000):
# 	path = base_path + 'cat.' + str(i) + '.jpg'
# 	# print(path)
# 	img = Image.open(path)
# 	img = img.resize((128, 128))
# 	img = np.asarray(img)

# 	data.append(img)

# data = np.reshape(data, [-1, 128*128*3])
# print(len(data))
# model

n_epochs = 30 # 30 before
# n_examples = 5000
batch_size = 128

data_ph = tf.placeholder('float', [None, 28*28], name = 'data_ph')
output_ph = tf.placeholder('float', [None, 28*28], name = 'output_ph')
learning_rate = tf.placeholder('float', [], name = 'learning_rate_ph')

weights = {
	'hl1': tf.Variable(tf.random_normal([28*28, 250])),
	# 'hl2': tf.Variable(tf.random_normal([500, 200])),
	'hl2': tf.Variable(tf.random_normal([250, 50])), # middle layer
	'hl3': tf.Variable(tf.random_normal([50, 250])),
	# 'hl5': tf.Variable(tf.random_normal([200, 500])),
	'ol': tf.Variable(tf.random_normal([250, 28*28]))
}

biases = {
	'hl1': tf.Variable(tf.random_normal([250])),
	# 'hl2': tf.Variable(tf.random_normal([200])),
	'hl2': tf.Variable(tf.random_normal([50])),
	# 'hl4': tf.Variable(tf.random_normal([200])),
	'hl3': tf.Variable(tf.random_normal([250])),
	'ol': tf.Variable(tf.random_normal([28*28]))	
}

hl1 = tf.nn.sigmoid(tf.add(tf.matmul(data_ph, weights['hl1']), biases['hl1']), name = 'hl1')
hl2 = tf.nn.sigmoid(tf.add(tf.matmul(hl1, weights['hl2']), biases['hl2']), name = 'hl2')
hl3 = tf.nn.sigmoid(tf.add(tf.matmul(hl2, weights['hl3']), biases['hl3']), name = 'hl3')
# hl4 = tf.nn.relu(tf.add(tf.matmul(hl3, weights['hl4']), biases['hl4']), name = 'hl4')
# hl5 = tf.nn.relu(tf.add(tf.matmul(hl4, weights['hl5']), biases['hl5']), name = 'hl5')
ol = tf.nn.sigmoid(tf.add(tf.matmul(hl1, weights['ol']), biases['ol']), name = 'ol')


loss = tf.reduce_mean((ol - output_ph)**2, name = 'loss')
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# saver = tf.train.import_meta_graph("E:/workspace_py/saved_models/autoencoder/autoencoder-1.ckpt.meta")
# saver.restore(sess, tf.train.latest_checkpoint('E:/workspace_py/saved_models/autoencoder/autoencoder-mnist'))

# err = 999999 # infinity
for epoch in range(n_epochs):
	ptr = 0
	for iteration in range(int(mnist.train.num_examples/batch_size)):

		epoch_x, epoch_y = mnist.train.next_batch(batch_size)
		# epoch_x = data[ptr : ptr + batch_size]
		# epoch_y = data[ptr : ptr + batch_size]
		# ptr += batch_size
		# if err < 12500:
		# 	_, err = sess.run([train, loss], feed_dict={data_ph: epoch_x, output_ph: epoch_y, learning_rate: 1.0})
		# else:
		_, err = sess.run([train, loss], feed_dict={data_ph: epoch_x, output_ph: epoch_x, learning_rate: 0.01})

	print("Loss @ epoch ", str(epoch), " = ", err)
	save_path = saver.save(sess, "E:/workspace_py/saved_models/autoencoder/autoencoder-mnist/autoencoder-1.ckpt")

# saver = tf.train.Saver()
prediction = sess.run(ol, feed_dict={data_ph: [mnist.train.images[0]]})
print("prediction: ", prediction)

import matplotlib.pyplot as plt

plt.imshow(np.reshape(prediction, [28, 28]))
plt.show()
sess.close()
