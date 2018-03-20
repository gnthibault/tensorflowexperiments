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
# max to keep means that we only keep the k last explicitly saved  models
# keep_checkpoint means that in addition of that, we keep track of models every
# n hours, and those are not accounted at all in the max_to_keep option
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)

#One can also on add ops to save and restore only `v2` using the name "v2"
saver2 = tf.train.Saver({"v2": v2})
#Or one can save with the default anme but only a suibset of the variables
saver3 = tf.train.Saver([v1,v2])

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)

    for i in range(40):
      # Do some work with the model.
      inc_v1.op.run()
      dec_v2.op.run()

      # Save the model + variables to disc. returns path
      save_path = saver.save(sess, "./ckpt/my_model", global_step=i)
      #The following files will be created:
      #my_model-0.index
      #my_model-0.meta
      #my_model-0.data-00000-of-00001
      #checkpoint
      print("Model saved in path: {}".format(save_path))

# Now reload everything:
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
save_path = tf.train.latest_checkpoint('./ckpt')

# Either create the void saver for restoring model
saver = tf.train.Saver()
# Or create the saver to specifically import the model
#print('Restoring from file {}.meta'.format(save_path))
#saver = tf.train.import_meta_graph('{}.meta'.format(save_path))


with tf.Session() as sess:

    saver.restore(sess, save_path)
    print("Model restored.")

    # Check the values of the variables
    print('v1 : {}'.format(v1.eval()))
    print('v2 : {}'.format(v2.eval()))
