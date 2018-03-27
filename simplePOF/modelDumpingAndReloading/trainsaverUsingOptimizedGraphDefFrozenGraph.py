# generic stuff
import os
import shutil
import sys

# tensorflow stuff
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# Math stuff
import numpy as np

#plot stuff
import matplotlib.pyplot as plt

class Test():

    def __init__(self):
    
        # define a model builder to save inference model
        self.model_ckpt_directory = './ckpt/'
        self.model_ckpt_path = './ckpt/my_model'
        self.graph_base_file_name = 'my_graph.pb'
        self.output_frozen_graph_base_name = 'frozen_'
        self.output_optimized_graph_base_name = 'optimized_'
        self.viz_dir = 'tensorboard'
        self.use_binary_export = True

        # main graph (may not always be populated)
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

        # Tensor/op name dedicated to frozen model restoration
        self.restore_op_name = "save/restore_all"
        self.frozen_filename = "save/Const:0"

    @property
    def graph_file_name(self):
        if self.use_binary_export:
            return self.graph_base_file_name
        else:        
            return self.graph_base_file_name+'txt'

    @property
    def output_frozen_graph_name(self):
        return os.path.join(self.model_ckpt_directory,
            self.output_frozen_graph_base_name+self.graph_file_name)

    @property
    def output_optimized_graph_name(self):
        return os.path.join(self.model_ckpt_directory,
            self.output_optimized_graph_base_name+self.graph_file_name)

    def initGraph(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as graph:
            with tf.name_scope('inputs'):
                x_placeholder = tf.placeholder(shape=[None],
                                               dtype=tf.float32,
                                               name=self.x_placeholder)
                # I had to create an op for the input, otherwise the
                # graph optimizing engine doesn't want to treat an input
                # that is a tensor
                x_placeholder = tf.identity(x_placeholder,
                                            name='x')
                print('Here is the name of the op {}'.format(
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
                                 name=self.graph_file_name,
                                 as_text=not self.use_binary_export)
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

            # We store the graph so we can view it with tboard
            trainWriter = tf.summary.FileWriter(
                os.path.join(self.model_ckpt_directory, self.viz_dir),
                graph=sess.graph)

            if step % 20 == 0:
                test_loss_val = sess.run([test_loss],
                                         feed_dict=te_feed_dict)
                print('step {}, test loss is {}'.format(
                      step, test_loss_val))
                # Once every 20 iterations, we save a checkpoint
                saveCkpt(step)
        #Here we are supposed to save the last ckpt
        saveCkpt(nb_iter-1)
        trainWriter.close()

    def initializeLinearRegressionTraining(self, xtr, ytr, xte, yte):
        """ Assuming the model is y = a x + b
        """
        with tf.Session(graph=self.graph) as sess:
            init_op = tf.global_variables_initializer()
            # Initialize variables
            sess.run(init_op)
            self.launchTrainingLoop(101, xtr, ytr, xte, yte, sess)
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

    def exportFrozenGraphForInference(self, clear_devices=True):

        # Freeze the graph
        input_graph_path = os.path.join(self.model_ckpt_directory,
                                        self.graph_file_name)
        ckpt_path = tf.train.latest_checkpoint(self.model_ckpt_directory)
        input_saver_def_path = ""
        output_node_name = "inference_model/"+self.y_inference

        print('Freezing graph {}'.format(input_graph_path))
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  self.use_binary_export, ckpt_path,
                                  output_node_name, self.restore_op_name,
                                  self.frozen_filename,
                                  self.output_frozen_graph_name,
                                  clear_devices, "")

    def optimizeFrozenNetwork(self):
        # Optimize for inference

        # First: load graphdef from protobuf file
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(self.output_frozen_graph_name, 'rb') as f:
            data = f.read()
        input_graph_def.ParseFromString(data)

        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            # array of input node(s), has to be op, no tensor
            ['inputs/x'], 
            # an array of output nodes
            ["inference_model/"+self.y_inference], 
            tf.float32.as_datatype_enum)

        # Save the optimized graph
        with tf.gfile.FastGFile(self.output_optimized_graph_name, "w") as f:
            f.write(output_graph_def.SerializeToString())

    def inferFromFrozenGraph(self, x):
        # First: load graphdef from protobuf file
        graph_def = tf.GraphDef()
        with tf.gfile.Open(self.output_optimized_graph_name, 'rb') as f:
            data = f.read()
        graph_def.ParseFromString(data)

        # loading model from file
        with tf.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def)
            input_node = sess.graph.get_tensor_by_name(
                'import/inputs/x'+':0')
            output_node = sess.graph.get_tensor_by_name(
                'import/inference_model/'+self.y_inference+':0')

            # run inference
            y = sess.run(output_node, feed_dict={input_node: x})

            #plot stuff
            plt.plot(x,y,'y.')

    def make_noisy_data(a=0.1, b=0.3, n=100):
        x = np.random.rand(n).astype(np.float32)
        noise = np.random.normal(scale=0.01, size=len(x))
        y = a * x + b + noise
        return np.float32(x), np.float32(y)

    def showPlot():
        plt.show()

if __name__ == '__main__':
    t = Test()
    t.initGraph()
    x_train, y_train = Test.make_noisy_data()
    x_test, y_test = Test.make_noisy_data()
    t.initializeLinearRegressionTraining(x_train, y_train,x_test, y_test)
    t.exportFrozenGraphForInference()
    t.optimizeFrozenNetwork()
    del t
    t = Test()
    t.inferFromFrozenGraph(x_test)
    Test.showPlot()
