# generic stuff
import os
import shutil
import sys

# tensorflow stuff
import tensorflow as tf
from tensorflow.python.saved_model import simple_save
from tensorflow.python.platform import gfile

# Math stuff
import numpy as np

#plot stuff
import matplotlib.pyplot as plt

class Test():

    def __init__(self):
    
        # define a model builder to save inference model
        self.model_ckpt_directory = './ckpt/'
        self.model_ckpt_path = './ckpt/my_model'
        self.graphFileName = 'my_graph.pb'
        self.graph = None

        # Tensor names decided in advance    
        self.x_placeholder = 'x_placeholder'
        self.y_placeholder = 'y_placeholder'
        self.a = 'weight'
        self.b = 'bias'
        self.y_train = 'y_train'
        self.y_inference = 'output_node'
        self.train_loss = 'train_loss'
        self.train = 'train'
        self.test_loss = 'test_loss'

    def initGraph(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as graph:
            with tf.name_scope('inputs'):
                x_placeholder = tf.placeholder(shape=[None],
                                               dtype=tf.float32,
                                               name=self.x_placeholder)
                print('Here is the name of the tensor {}'.format(
                    x_placeholder.name))
                y_placeholder = tf.placeholder(shape=[None],
                                               dtype=tf.float32,
                                               name=self.y_placeholder)
            with tf.name_scope('train_model'):
                a = tf.Variable(tf.random_normal([1]), name=self.a)
                b = tf.Variable(tf.random_normal([1]), name=self.b)
                y_train = tf.identity(a*x_placeholder+b,
                                      name=self.y_train)

            # In some cases, the inference model may be different
            with tf.name_scope('inference_model'):
                y_inference = tf.identity(a*x_placeholder+b,
                                          name=self.y_inference)

            LEARNING_RATE = 0.5
            with tf.name_scope('training'):
                with tf.name_scope('loss'):
                    train_loss = tf.reduce_mean(
                        tf.square(y_train - y_placeholder),
                        name=self.train_loss)
                with tf.name_scope('optimizer'):
                    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
                    train = optimizer.minimize(train_loss, name=self.train)
            # test loss may be different
            with tf.name_scope('test'):
                with tf.name_scope('loss'):
                    test_loss = tf.reduce_mean(
                               tf.square(y_inference - y_placeholder),
                               name=self.test_loss)

            # Add ops to save and restore all the variables.
            # max to keep means that we only keep the k last explicitly
            # saved models.
            # keep_checkpoint means that in addition of that, we keep track of
            # models every n hours, but those are not accounted at all in the
            # max_to_keep option
            self.ckpt_saver = tf.train.Saver(max_to_keep=1,
                                             keep_checkpoint_every_n_hours=2)

            # Also store the graph definition
            tf.train.write_graph(self.graph, logdir=self.model_ckpt_directory,
                                 name=self.graphFileName, as_text=False)

    def launchTrainingLoop(self, nb_iter, xtr, ytr, xte, yte, sess):
        graph = tf.get_default_graph()

        # get tensors by name
        x_placeholder = graph.get_tensor_by_name(
            'inputs/'+self.x_placeholder+':0')
        y_placeholder = graph.get_tensor_by_name(
            'inputs/'+self.y_placeholder+':0')
        train = graph.get_operation_by_name(
            'training/optimizer/'+self.train)
        test_loss = graph.get_tensor_by_name(
            'test/loss/'+self.test_loss+':0')

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
            init_op = tf.global_variables_initializer()
            # Initialize variables
            sess.run(init_op)
            self.launchTrainingLoop(81, xtr, ytr, xte, yte, sess)

    def restartLinearRegression(self, xtr, ytr, xte, yte):
        ckpt_path = tf.train.latest_checkpoint(self.model_ckpt_directory)
        print('Found latest ckpt file: {}'.format(ckpt_path))

        #other possibility, reload graph as well !!!
        graphFilePath = os.path.join(self.model_ckpt_directory,
                                     self.graphFileName)
        with gfile.FastGFile(graphFilePath,'rb') as f:
            print('Now parsing file {}'.format(graphFilePath))
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            print('Before graph import {}'.format(
                graph.get_all_collection_keys()))
            tf.import_graph_def(graph_def)
            print('After graph import {}'.format(
                graph.get_all_collection_keys()))

            with tf.Session(graph=graph) as sess:
                #This will load the graphdef into the default graph of the Session
                #tf.import_graph_def(graph_def)
                #self.graph = sess.graph
                print('After graph load in session {}'.format(
                    sess.graph.get_all_collection_keys()))
                # We don't default initialize variables, but restore them
                tf.train.Saver().restore(sess, ckpt_path)
                print("Model restored.")

                self.launchTrainingLoop(121, xtr, ytr, xte, yte, sess)

                # get final training results
                a = sess.graph.get_tensor_by_name('train_model/'+self.a+':0')
                aval = sess.run(a)
                b = sess.graph.get_tensor_by_name('train_model/'+self.b+':0')
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
    t.restartLinearRegression(x_train, y_train,x_test, y_test)
    t.showPlot()
