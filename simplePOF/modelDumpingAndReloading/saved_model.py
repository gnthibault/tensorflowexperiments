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
        self.model_directory = './ckpt/model/'
        try:
            shutil.rmtree(self.model_directory)
        except:
            pass
        self.saved_model_builder = tf.saved_model.builder.SavedModelBuilder(
            self.model_directory)

        self.graph = tf.Graph()
        with self.graph.as_default() as graph:
            with tf.name_scope('inputs'):
                self.x_placeholder = tf.placeholder(shape=[None],
                                                    dtype=tf.float32, name='x')
                self.y_placeholder = tf.placeholder(shape=[None],
                                                    dtype=tf.float32, name='y')
            with tf.name_scope('train_model'):
                self.a = tf.Variable(tf.random_normal([1]), name='weight')
                self.b = tf.Variable(tf.random_normal([1]), name='bias')
                self.y_train = self.a*self.x_placeholder+self.b
            # In some cases, the inference model may be different
            with tf.name_scope('inference_model'):
                self.y_inference = self.a*self.x_placeholder+self.b
                self.y_inference = tf.identity(self.y_inference,
                                               name='output_node')

    def trainLinearRegression(self, xtr, ytr, xte, yte):
        """ Assuming the model is y = a x + b
        """
        LEARNING_RATE = 0.5
        with self.graph.as_default() as graph:
            with tf.name_scope('training'):
                with tf.name_scope('loss'):
                    train_loss = tf.reduce_mean(
                        tf.square(self.y_train - self.y_placeholder))
                with tf.name_scope('optimizer'):
                    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
                    train = optimizer.minimize(train_loss) 
            # test loss may be different
            with tf.name_scope('test'):
                with tf.name_scope('loss'):
                    test_loss = tf.reduce_mean(
                               tf.square(self.y_inference - self.y_placeholder))

            # Set up the signature for Predict with input and output tensor
            # specification.
            predict_signature_def = self._build_regression_signature(
                self.x_placeholder, self.y_inference)
            signature_def_map = {'regress_x_to_y': predict_signature_def}

            with tf.Session() as sess:
                # Initialize variables
                sess.run(tf.global_variables_initializer())
                TRAIN_STEPS = 201

                for step in range(TRAIN_STEPS):
                    sess.run([train], 
                             feed_dict={self.x_placeholder: xtr, 
                                        self.y_placeholder: ytr})
                    if step % 20 == 0:
                        test_loss_val = sess.run([test_loss],
                             feed_dict={self.x_placeholder: xte, 
                                        self.y_placeholder: yte})
                        print('step {}, test loss is {}'.format(
                              step, test_loss_val))

                # get final training results
                a = sess.run(self.a)
                b = sess.run(self.b)

                # now that variables are initialized, you can save inference
                self.saved_model_builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map=signature_def_map)
                self.saved_model_builder.save(as_text=True)
             
            # draw stuff to show that it works
            minx=np.min(np.concatenate((xtr,xte)))
            maxx=np.max(np.concatenate((xtr,xte)))
            xref=np.linspace(minx,maxx,100)
            plt.figure(0)
            plt.plot(xref, a*xref+b, 'r.')
            plt.plot(xtr, ytr, 'b.')
            plt.plot(xte, yte, 'g.')

    def inferLinearRegression(self, x):
        # loading model from file
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       self.model_directory)
            graph = tf.get_default_graph()
            input_node = graph.get_tensor_by_name(
                "inputs/x:0")
            output_node = graph.get_tensor_by_name(
                "inference_model/output_node:0")

            # run inference
            y = sess.run(output_node, feed_dict={input_node: x})

            #plot stuff
            plt.plot(x,y,'y.')

    def make_noisy_data(a=0.1, b=0.3, n=100):
        x = np.random.rand(n).astype(np.float32)
        noise = np.random.normal(scale=0.01, size=len(x))
        y = a * x + b + noise
        return np.float32(x), np.float32(y)

    def _build_regression_signature(self, input_tensor, output_tensor):
        """Helper function for building a regression SignatureDef.
           Possible signature_constants keys are:
               CLASSIFY_INPUTS
               CLASSIFY_METHOD_NAME
               CLASSIFY_OUTPUT_CLASSES
               CLASSIFY_OUTPUT_SCORES
               DEFAULT_SERVING_SIGNATURE_DEF_KEY
               PREDICT_INPUTS
               PREDICT_METHOD_NAME
               PREDICT_OUTPUTS
               REGRESS_INPUTS
               REGRESS_METHOD_NAME
               REGRESS_OUTPUTS
        """
        input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
        signature_inputs = {
            tf.saved_model.signature_constants.REGRESS_INPUTS: input_tensor_info
        }
        output_tensor_info = tf.saved_model.utils.build_tensor_info(output_tensor)
        signature_outputs = {
            tf.saved_model.signature_constants.REGRESS_OUTPUTS: output_tensor_info
        }
        return tf.saved_model.signature_def_utils.build_signature_def(
            signature_inputs, signature_outputs,
            tf.saved_model.signature_constants.REGRESS_METHOD_NAME)

    def showPlot(self):
        plt.show()

if __name__ == '__main__':
    t = Test()
    x_train, y_train = Test.make_noisy_data()
    x_test, y_test = Test.make_noisy_data()
    t.trainLinearRegression(x_train, y_train,x_test, y_test)
    t.inferLinearRegression(x_test)
    t.showPlot()
