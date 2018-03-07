import numpy as np
import tensorflow as tf

with tf.name_scope("MyLayerGroup"):
    with tf.name_scope('Layer_0'):
        a0 = tf.placeholder(dtype=tf.float32, shape=[None], name='in0')
        a1 = tf.placeholder(dtype=tf.float32, shape=[None], name='in1')
    with tf.name_scope('Layer_1'):
        #If you don't specify dtype=tf.float32, it won't work
        b0 = tf.constant(1, dtype=tf.float32, name='bias1')
        tf.summary.scalar('constant bias', b0)
        b1 = a0
        b2 = tf.pow(a0, 2)
        b3 = tf.multiply(a0, a1)
        tf.summary.tensor_summary('cross term', b3)
        b4 = tf.pow(a1, 2)
        b5 = a1
    with tf.name_scope('Layer_2'):
        c0 = tf.add_n([b1, b2, b3, b4, b5])
        c0 = tf.multiply(b0,c0)

summary_nn = tf.summary.merge_all()

with tf.Session() as sess:
    trainWriter = tf.summary.FileWriter("OutputLogDirectory", graph=sess.graph)

    la0 = np.random.rand(5)
    la1 = np.random.rand(5)
  
    for step in range(10):

        # collect variable of interest + summary
        summary, val = sess.run([summary_nn, c0], feed_dict={a0:la0*step,a1:la1*step})
        print('Value of c0 is {}'.format(val))

        # also collect train metadata
        run_metadata = tf.RunMetadata()
        trainWriter.add_run_metadata(run_metadata, '{}'.format(step)) 
        trainWriter.add_summary(summary,step)

    trainWriter.close()


#At the end of the run, you can lauch
#tensorboard --logdir=OutputLogDirectory

