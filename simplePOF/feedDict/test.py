import tensorflow as tf

a0 = tf.placeholder_with_default(10, shape=[], name='in0')
a1 = tf.placeholder_with_default(-10, shape=[], name='in1')


b0 = tf.constant(1, name='bias1')
b1 = a0
b2 = tf.pow(a0, 2)
b3 = tf.multiply(a0, a1)
b4 = tf.pow(a1, 2)
b5 = a1

c0 = tf.add_n([b0, b1, b2, b3, b4, b5])

with tf.Session() as sess:
    val = sess.run(c0, feed_dict={a0:11,a1:12})
    print('Value of c0 is {}'.format(val))
    #Cannot use the name of the placeholder, directly, need to suffix with index
    val = sess.run(c0, feed_dict={'in0:0':13,'in1:0':14})
    print('Value of c0 is {}'.format(val))
    val = sess.run(c0)
    print('Value of c0 is {}'.format(val))

