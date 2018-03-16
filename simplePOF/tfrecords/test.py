# tensorflow stuff
import tensorflow as tf

# Image stuff
import scipy.misc as scm


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

#Here is the best way to store data for training
# This method is especially suited because it writes data in a binary format
# to a file.
# During training, part of this file can be retrieved without loading the
# entire file in ram, and there is no need to decode data


# Declare a writer
tfrecord_name = 'outputdir/filename.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecord_name)

# generates some images
nsamples = 10

# Please take take of using a random sample order here
for i in range(nsamples):
    # Usually one open individual files there
    feature = scm.face()
    label = scm.face()[:,:,1]>100

    # then convert to strings
    raw_feature = feature.tostring()
    raw_label = label.tostring()

    # tf.train.Example() call instantiates a new protocol buffer,
    # and fills in some of its fields

    # data should be converted to either tf.train.Int64List, tf.train.BytesList,
    # or  tf.train.FloatList

    colorspace = 'RGB'
    width = feature.shape[1]
    height = feature.shape[0]
    channels = feature.shape[2]
    quality = 0.1
    comment = 'This is a comment'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(raw_feature),
        'image/label': _bytes_feature(raw_label),
        'image/width': _int64_feature(width),
        'image/height': _int64_feature(height),
        'image/channels': _int64_feature(channels),
        'image/colorspace': _bytes_feature(colorspace.encode()),
        'image/info/quality': _float_feature(quality),
        'image/info/comment': _bytes_feature(comment.encode())}))
        
    #Then, we serialize the protocol buffer to a string and write it to a tfr
    writer.write(example.SerializeToString())

writer.close()

# Now read
for sample in tf.python_io.tf_record_iterator(tfrecord_name):
  print('type is {}'.format(type(sample)))
