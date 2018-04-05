#pre-requisite
#sudo apt install libprotobuf-dev protobuf-compiler
#pip install onnx-tf


# Generic stuff
import urllib.request

# Plot stuff
import matplotlib.pyplot as plt

# numerical stuff
import numpy as np

# Image stuff
import PIL
import scipy.misc as scm

# DL stuff
import tensorflow as tf
import onnx
import onnx_tf.backend


#Download the model
base_url = 'https://raw.githubusercontent.com/onnx/tutorials/master/tutorials/assets/'
model_name = 'super_resolution.onnx'
urllib.request.urlretrieve(base_url+model_name, model_name)


# Load the model, model is of type onnx_pb2.ModelProto
model = onnx.load('super_resolution.onnx')
# tf_ref is of type onnx_tf.backend_rep.TensorflowRep
tf_rep = onnx_tf.backend.prepare(model)


#Now we have tf_rep, which is a python class containing three members: predict_net, input_dict, and uninitialized

# Network is of type onnx_tf.tf_net.TensorflowNet, which contains the following
# attributes:  external_input,  op, external_output output, name, output_dict
print('The network: {}'.format(tf_rep.predict_net))
print('-----')

#print the list of operators in the network:
print('List of operators in the network {}'.format(tf_rep.predict_net.op))
print('-----')

#You can get a dictionnary that maps indices to tensors
print('Dictionary from indices to tensors: {}'.format(
      tf_rep.predict_net.output_dict))
#and also get the index of the output tensor from this dictionary
print('The particular index of the output tensor: {}'.format(
      tf_rep.predict_net.external_output))

# input_dict is a python dict that maps names to tensor and that you may fill-in
# in order to run the network
print('Now here is the input dictionary that should be used to '
      'define tensors {}'.format(tf_rep.input_dict))
print('-----')

# uninitialized is a python list of the keys inside input_dict that should be
# initialized (the other one already have a value)
print('The key inside the input_dict, of the tensor that is mandatory '
      'and should be initialized {}'.format(tf_rep.uninitialized))

# Now load the data
img = PIL.Image.fromarray(scm.face())
img = img.crop((0,0,768,768))
img = img.resize((224, 224))
img_ycbcr = img.convert("YCbCr")
img_y, img_cb, img_cr = img_ycbcr.split()
img_input = np.asarray(img_y, dtype=np.float32)[np.newaxis, np.newaxis, :, :]

# one can now run the network, providing the 'feed dictionary' of uninitialized
# keys
#output is of type onnx.backend.base.Outputs which is a kind of readonly list
output = tf_rep.run({tf_rep.uninitialized[0]:img_input})
img_output = output[0]

# Now format the output and print to screen
img_out_y = PIL.Image.fromarray(np.uint8(img_output[0, 0, :, :].clip(0, 255)), mode='L')
result_img = PIL.Image.merge("YCbCr", [
  img_out_y,
  img_cb.resize(img_out_y.size, PIL.Image.BICUBIC),
  img_cr.resize(img_out_y.size, PIL.Image.BICUBIC),
]).convert("RGB")

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(result_img)
plt.show()
