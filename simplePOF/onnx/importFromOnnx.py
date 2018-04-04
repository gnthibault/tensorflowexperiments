

# DL stuff
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# Load the model
model = onnx.load('assets/super_resolution.onnx')
tf_rep = prepare(model)


#Now we have tf_rep, which is a python class containing three members: predict_net, input_dict, and uninitialized
print(tf_rep.predict_net)
print('-----')
print(tf_rep.input_dict)
print('-----')
print(tf_rep.uninitialized)
