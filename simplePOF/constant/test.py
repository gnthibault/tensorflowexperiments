import tensorflow as tf


x = tf.constant(value=[10,20,30], name='x')
y = tf.Variable(x+1, name='y')

initializer = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(initializer)
  print('Final value of y is {}'.format(sess.run(y)))
