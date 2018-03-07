import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf


print('Initializing test')


image = scipy.misc.face()
#x = tf.constant(value=[10,20,30], name='x')
y = tf.Variable(image, name='y')

initializer = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(initializer)
  #print('Final value of y is {}'.format(sess.run(y)))
  plt.imshow(sess.run(y))
  plt.show()
