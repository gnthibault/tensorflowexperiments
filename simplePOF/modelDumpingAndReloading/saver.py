import tensorflow as tf


# Create some variables.
v1 = tf.get_variable('v1', shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)

# New nodes
inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#One can also on add ops to save and restore only `v2` using the name "v2"
saver2 = tf.train.Saver({"v2": v2})
#Or one can save with the default anme but only a suibset of the variables
saver3 = tf.train.Saver([v1,v2])

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)

    for i in range(10):
      # Do some work with the model.
      inc_v1.op.run()
      dec_v2.op.run()

      if i==0:
          # Save the variables to disk.
          save_path = saver.save(sess, "./ckpt/my_model", global_step=i)
          print("Model saved in path: %s" % save_path)
          #The following files will be created:
          #my_model-0.index
          #my_model-0.meta
          #my_model-0.data-00000-of-00001
          #checkpoint
      else:
          #After first save, no need to store meta graph anymore
          save_path = saver.save(sess, "./ckpt/my_model", global_step=i,
              write_meta_graph=False)





# Now reload everything:
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./ckpt/my_model")
    print("Model restored.")
    # Check the values of the variables
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())
