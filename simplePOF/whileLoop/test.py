import tensorflow as tf

nb_iter = tf.constant(value=0)
#This solution does not work at all
#nb_iter = tf.get_variable('nb_iter', shape=(1), dtype=tf.int32, trainable=False)
i = tf.get_variable('i', shape=(), trainable=False,
                     initializer=tf.zeros_initializer(), dtype=nb_iter.dtype)

loop_condition = lambda i: tf.less(i, nb_iter)
def loop_body(i):
    print('Another iteration')
    return [tf.add(i, 1)]

i = tf.while_loop(loop_condition, loop_body, [i])

initializer_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer_op)
    res = sess.run(i)
    print('res is now {}'.format(res))

