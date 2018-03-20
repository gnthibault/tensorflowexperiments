# generic stuff
import sys

# tensorflow stuff
import tensorflow as tf
from tensorflow.python.saved_model import simple_save

# Math stuff
import numpy as np

#plot stuff
import matplotlib.pyplot as plt

class Test():

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as graph:
            with tf.name_scope('input'):
                self.x_placeholder = tf.placeholder(shape=[None],
                                                    dtype=tf.float32, name='x')
                self.y_placeholder = tf.placeholder(shape=[None],
                                                    dtype=tf.float32, name='y')
            with tf.name_scope('train_model'):
                self.a = tf.Variable(tf.random_normal([1]), name='weight')
                self.b = tf.Variable(tf.random_normal([1]), name='bias')
                self.y_train = self.a*self.x_placeholder+self.b
            # In some cases, the inference model may be different
            with tf.name_scope('test_model'):
                self.y_test = self.a*self.x_placeholder+self.b

    def linearRegression(self, xtr, ytr, xte, yte):
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
                               tf.square(self.y_test - self.y_placeholder))
            with tf.Session() as sess:
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

                # Final training results
                a = sess.run(self.a)
                b = sess.run(self.b)
        # Draw result
        minx=np.min(np.concatenate((xtr,xte)))
        maxx=np.max(np.concatenate((xtr,xte)))
        xref=np.linspace(minx,maxx,100)
        plt.figure(0)
        plt.plot(xref, a*xref+b, 'r.')
        plt.plot(xtr, ytr, 'b.')
        plt.plot(xte, yte, 'g.')
        plt.show()

    def make_noisy_data(a=0.1, b=0.3, n=100):
        x = np.random.rand(n).astype(np.float32)
        noise = np.random.normal(scale=0.01, size=len(x))
        y = a * x + b + noise
        return np.float32(x), np.float32(y)

# Declare a writer
#tfrecord_name = 'outputdir/filename.tfrecords'

#with tf.get_default_graph() as graph:
#    x = tf.constant(value=[10,20], name='constante')
#    y = tf.Variable(x+1, name='input')
#    z = tf.Variable(y+1, name='output')


#with tf.Session() as sess:

    # Save the variables to disk.
#    model_dir = "./ckpt/"
#    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)

#    input_tensor = tf.get_default_graph().get_tensor_by_name("input:0")
#    output_tensor = tf.get_default_graph().get_tensor_by_name("output:0")
    #tensor_info_in = tf.saved_model.utils.build_tensor_info(input_tensor)
    #tensor_info_out = tf.saved_model.utils.build_tensor_info(output_tensor)
    #simple_save.simple_save(sess, )
    #    tf.saved_model.signature_def_utils.build_signature_def(
    #        inputs={'input': tensor_info_in},
    #        outputs={'output': tensor_info_out},
    #        method_name=tf.saved_model.
    #        signature_constants.PREDICT_METHOD_NAME)
    #builder.add_meta_graph_and_variables(
    #    sess,
    #    [tf.saved_model.tag_constants.SERVING],
    #    signature_def_map={'predict_images':
    #                           prediction_signature})
    #builder.save()
if __name__ == '__main__':
    t = Test()
    x_train, y_train = Test.make_noisy_data()
    x_test, y_test = Test.make_noisy_data()
    t.linearRegression(x_train, y_train,x_test, y_test)
