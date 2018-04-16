import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

#truth value for the linear model
k_true = [[1, -1], [2, -2], [3, -3]]
b_true = [-5, 5]
num_examples = 128


with tf.Session() as sess:
    # Input place holders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')

    # Define model, architecture, loss and training
    dense_layer = tf.keras.layers.Dense(2, use_bias=True)
    y_hat = dense_layer(x)
    loss = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(y, y_hat), name='loss')
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Initializ model variables
    sess.run(tf.global_variables_initializer())

    sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:7000')
    for step in range(50):
        #Generate synthetic traininf data
        xs = np.random.randn(num_examples, 3)
        ys = np.matmul(xs, k_true) + b_true

        loss_val, _ = sess.run([loss, train_op], feed_dict={x:xs, y:ys})
        print('Iteration {}, loss: {}'.format(step, loss_val))

#At the end of the run, you can lauch
#tensorboard --logdir ./output/ --debugger_port 7000
