# generic stuff
import sys
import shutil

# tensorflow stuff
import tensorflow as tf
from tensorflow.python.saved_model import simple_save
from tensorflow.python.lib.io import file_io

# Math stuff
import numpy as np

#plot stuff
import matplotlib.pyplot as plt

class Test():

    def __init__(self):
    
        # define a model builder to save inference model
        self.model_ckpt_directory = './ckpt/'
        self.model_ckpt_path = './ckpt/my_model'


    def initGraph(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as graph:
            with tf.name_scope('inputs'):
                x_placeholder = tf.placeholder(shape=[None],
                                               dtype=tf.float32, name='x')
                self.x_placeholder = x_placeholder.name
                y_placeholder = tf.placeholder(shape=[None],
                                               dtype=tf.float32, name='y')
                self.y_placeholder = y_placeholder.name
            with tf.name_scope('train_model'):
                a = tf.Variable(tf.random_normal([1]), name='weight')
                self.a = a.name
                b = tf.Variable(tf.random_normal([1]), name='bias')
                self.b = b.name
                y_train = a*x_placeholder+b
                self.y_train = y_train.name
            # In some cases, the inference model may be different
            with tf.name_scope('inference_model'):
                y_inference = a*x_placeholder+b
                y_inference = tf.identity(y_inference, name='output_node')
                self.y_inference = y_inference.name

            LEARNING_RATE = 0.5
            with tf.name_scope('training'):
                with tf.name_scope('loss'):
                   train_loss = tf.reduce_mean(
                        tf.square(y_train - y_placeholder))
                   self.train_loss = train_loss.name
                with tf.name_scope('optimizer'):
                    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
                    train = optimizer.minimize(train_loss)
                    self.train = train.name
            # test loss may be different
            with tf.name_scope('test'):
                with tf.name_scope('loss'):
                    test_loss = tf.reduce_mean(
                               tf.square(y_inference - y_placeholder))
                    self.test_loss = test_loss.name

            # Add ops to save and restore all the variables.
            # max to keep means that we only keep the k last explicitly
            # saved models.
            # keep_checkpoint means that in addition of that, we keep track of
            # models every n hours, but those are not accounted at all in the
            # max_to_keep option
            self.ckpt_saver = tf.train.Saver(max_to_keep=1,
                                             keep_checkpoint_every_n_hours=2)

    def launchTrainingLoop(self, nb_iter, xtr, ytr, xte, yte, sess):
        graph = tf.get_default_graph()

        # get tensors by name
        x_placeholder = graph.get_tensor_by_name(
            self.x_placeholder)
        y_placeholder = graph.get_tensor_by_name(
            self.y_placeholder)
        train = graph.get_operation_by_name(
            self.train)
        test_loss = graph.get_tensor_by_name(
            self.test_loss)

        tr_feed_dict = {x_placeholder: xtr, 
                     y_placeholder: ytr}
        te_feed_dict = {x_placeholder: xte, 
                     y_placeholder: yte}

        def saveCkpt(step):
            save_path = self.ckpt_saver.save(sess,
                                             self.model_ckpt_path,
                                             global_step=step)

        for step in range(nb_iter):
            sess.run([train], feed_dict=tr_feed_dict)
            if step % 20 == 0:
                test_loss_val = sess.run([test_loss],
                                         feed_dict=te_feed_dict)
                print('step {}, test loss is {}'.format(
                      step, test_loss_val))
                # Once every 20 iterations, we save a checkpoint
                saveCkpt(step)
        #Here we are supposed to save the last ckpt
        saveCkpt(nb_iter-1)

    def initializeLinearRegressionTraining(self, xtr, ytr, xte, yte):
        """ Assuming the model is y = a x + b
        """
        with tf.Session(graph=self.graph) as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            self.launchTrainingLoop(81, xtr, ytr, xte, yte, sess)

    def restartLinearRegression(self, xtr, ytr, xte, yte):
        save_path = tf.train.latest_checkpoint(self.model_ckpt_directory)
        print('Found latest ckpt file: {}'.format(save_path))

        # loading model from file, if graph is note initialized,
        # the restore does not works! and says that graph is empty
        with tf.Session(graph=self.graph) as sess:
            # We don't default initialize variables, but restore them
            self.ckpt_saver.restore(sess, save_path)
            print("Model restored.")

            self.launchTrainingLoop(121, xtr, ytr, xte, yte, sess)

            # get final training results
            a = self.graph.get_tensor_by_name(self.a)
            aval = sess.run(a)
            b = self.graph.get_tensor_by_name(self.b)
            bval = sess.run(b)

        # draw stuff to show that it works
        minx=np.min(np.concatenate((xtr,xte)))
        maxx=np.max(np.concatenate((xtr,xte)))
        xref=np.linspace(minx,maxx,100)
        plt.figure(0)
        plt.plot(xref, aval*xref+bval, 'r.')
        plt.plot(xtr, ytr, 'b.')
        plt.plot(xte, yte, 'g.')

    def make_noisy_data(a=0.1, b=0.3, n=100):
        x = np.random.rand(n).astype(np.float32)
        noise = np.random.normal(scale=0.01, size=len(x))
        y = a * x + b + noise
        return np.float32(x), np.float32(y)

    def showPlot(self):
        plt.show()

if __name__ == '__main__':
    t = Test()
    t.initGraph()
    x_train, y_train = Test.make_noisy_data()
    x_test, y_test = Test.make_noisy_data()
    t.initializeLinearRegressionTraining(x_train, y_train,x_test, y_test)
    del t
    t = Test()
    t.initGraph()
    t.restartLinearRegression(x_train, y_train,x_test, y_test)
    t.showPlot()
