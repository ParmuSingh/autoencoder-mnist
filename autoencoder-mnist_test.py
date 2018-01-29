import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("E:/workspace_py/mnist/", one_hot=True)

sess = tf.Session()

saver = tf.train.import_meta_graph("E:/workspace_py/saved_models/autoencoder/autoencoder-mnist/autoencoder-1.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint('E:/workspace_py/saved_models/autoencoder-mnist/'))


# print(sess.run('w_ol:0'))

graph = tf.get_default_graph()

restored_placeholder_input = graph.get_tensor_by_name('data_ph:0')

restored_variable_w_ol = graph.get_tensor_by_name('w_ol:0')

# ptr = 0
# for epoch in range(len(feature_test)):
prediction = sess.run(restored_variable_w_ol, feed_dict={restored_placeholder_input: mnist.train.images[0]})
print("prediction: ", prediction)

import matplotlib.pyplot as plt

plt.imshow(np.reshape(prediction, [28, 28]))
plt.show()
sess.close()