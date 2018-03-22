# tensorflow stuff
import tensorflow as tf

# Math stuff
import numpy as np

# Image stuff
import scipy.misc as scm

# tracing/showing/verbose tools
import matplotlib.pyplot as plt

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_feature(value):
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
    feature = scm.face().astype(np.float32)
    label = (scm.face()[:,:,1]>100).astype(np.float32)

    # then convert to strings
    raw_feature = feature.tostring()
    raw_label = label.tostring()

    # tf.train.Example() call instantiates a new protocol buffer,
    # and fills in some of its fields
    # data should be converted to either tf.train.Int64List, tf.train.BytesList,
    # or  tf.train.FloatList (float32)
    colorspace = 'RGB'
    width = feature.shape[1]
    height = feature.shape[0]
    channels = feature.shape[2]
    quality = 0.1
    comment = 'This is a comment'

    example = tf.train.Example(features=tf.train.Features(feature={
        'train/image': _bytes_feature(raw_feature),
        'train/label': _bytes_feature(raw_label),
        'train/width': _int64_feature(width),
        'train/height': _int64_feature(height),
        'train/channels': _int64_feature(channels),
        'train/colorspace': _bytes_feature(colorspace.encode()),
        'train/info/quality': _float32_feature(quality),
        'train/info/comment': _bytes_feature(comment.encode())}))
        
    #Then, we serialize the protocol buffer to a string and write it to a tfr
    writer.write(example.SerializeToString())

writer.close()

# Now read
#for sample in tf.python_io.tf_record_iterator(tfrecord_name):
#  print('type is {}'.format(type(sample)))


with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string),
               'train/width': tf.FixedLenFeature([], tf.int64),
               'train/height': tf.FixedLenFeature([], tf.int64),
               'train/channels': tf.FixedLenFeature([], tf.int64),
               'train/colorspace': tf.FixedLenFeature([], tf.string),
               'train/info/quality': tf.FixedLenFeature([], tf.float32),
               'train/info/comment': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecord_name],
                                                    num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    label = tf.decode_raw(features['train/label'], tf.float32)

    # get info about images
    width = tf.cast(features['train/width'], tf.int32)
    height = tf.cast(features['train/height'], tf.int32)
    channels = tf.cast(features['train/channels'], tf.int32)

    # get some more info if needed
    comment = tf.cast(features['train/image'], tf.string)
    quality = tf.cast(features['train/image'], tf.float32)
    #print('Comment was: {} and quality was {}'.format(comment, quality))

    # Reshape image data into the original shape
    # aShape must be defined otherwise, get
    # ValueError: All shapes must be fully defined:
    #image = tf.reshape(image, [height, width, channels])
    image = tf.reshape(image, [768, 1024, 3])
    label = tf.reshape(label, [768, 1024])


    # reformat data if needed.
    
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=2,
                                            capacity=50,
                                            num_threads=1,
                                            min_after_dequeue=10)

    # the train shuffle batch generates its own ops for parallel computations:
    #This function adds the following to the current Graph:
    #A shuffling queue into which tensors from tensors are enqueued.
    #A dequeue_many operation to create batches from the queue.
    #A QueueRunner to QUEUE_RUNNER collection, to enqueue the tensors from
    # tensors.

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        plt.subplot(111)
        plt.imshow(img[0, ...])
        plt.title('image from tfrecord')
        plt.show()

    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
