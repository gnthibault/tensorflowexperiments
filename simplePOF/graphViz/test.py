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

summaryOp = tf.summary.merge_all()

with tf.Session() as sess:
    trainWriter = tf.summary.FileWriter("OutputLogDirectory/train", graph=sess.graph)
    testWriter = tf.summary.FileWriter("OutputLogDirectory", graph=sess.graph)

    la0 = np.random.rand(5)
    la1 = np.random.rand(5)
  
    for step in range(10):
        if step % 2 == 0:  # Record summaries and test-set accuracy
            summary, res = sess.run([summaryOp, c0],
                                    feed_dict={a0:la0, a1:la1})#test feed
            testWriter.add_summary(summary, step)
            print('Value of c0 is {}'.format(res))
        else:  # Record train set summaries, and train
            if step % 5 == 0:  # Record execution stats
                # collect variable of interest + summary + metadata
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, res = sess.run([summaryOp, c0],
                                        feed_dict={
                                                   a0:la0*step,
                                                   a1:la1*step},# train feed
                                        options=run_options,
                                        run_metadata=run_metadata)
                print('Value of c0 is {}'.format(res))

                # also collect train metadata
                trainWriter.add_run_metadata(run_metadata, 'step: {}'.format(step)) 
                trainWriter.add_summary(summary, step)
            else:  # Record a summary
                summary, _ = sess.run([summaryOp, c0],
                                      feed_dict={a0:la0, a1:la1}) # train feed
                trainWriter.add_summary(summary, step)


    trainWriter.close()
    testWriter.close()

#At the end of the run, you can lauch
#tensorboard --logdir=OutputLogDirectory

