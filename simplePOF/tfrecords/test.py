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

# Please take care of using a random sample order here
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
    quality = np.float32(0.1)
    comment = 'This is a comment'
    objects_number = 2
    bboxes = [0,0,256,256,512,512,768,768]
    bb_labels = [5,2]

    example = tf.train.Example(features=tf.train.Features(feature={
        # Image data part
        'train/image': _bytes_feature(raw_feature),
        'train/label': _bytes_feature(raw_label),

        # Image content informations
        'train/objects_number': _int64_feature(objects_number),
        'train/bboxes': _int64_feature(bboxes),
        'train/bb_labels': _int64_feature(bb_labels),

        # Image info part
        'train/width': _int64_feature(width),
        'train/height': _int64_feature(height),
        'train/channels': _int64_feature(channels),
        'train/colorspace': _bytes_feature(colorspace.encode()),

        # Miscellaneous image info
        'train/info/quality': _float32_feature(quality),
        'train/info/comment': _bytes_feature(comment.encode())}))
        
    #Then, we serialize the protocol buffer to a string and write it to a tfr
    writer.write(example.SerializeToString())

writer.close()

# Now read
#for sample in tf.python_io.tf_record_iterator(tfrecord_name):
#  print('type is {}'.format(type(sample)))

nb_iter = 5
with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string),
               'train/objects_number':tf.FixedLenFeature([], tf.int64),
               'train/bboxes': tf.VarLenFeature(tf.float32),
               'train/bb_labels': tf.VarLenFeature(tf.int64),
               'train/width': tf.FixedLenFeature([], tf.int64),
               'train/height': tf.FixedLenFeature([], tf.int64),
               'train/channels': tf.FixedLenFeature([], tf.int64),
               'train/colorspace': tf.FixedLenFeature([], tf.string),
               'train/info/quality': tf.FixedLenFeature([], tf.float32),
               'train/info/comment': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecord_name],
                                                    num_epochs=nb_iter)

    #Note: if num_epochs is not None, this function creates local counter epochs
    # Use local_variables_initializer() to initialize local variables.
    #Args:
    #-string_tensor: A 1-D string tensor with the strings to produce.
    #-num_epochs: An integer (optional). If specified, string_input_producer
    # produces each string from string_tensor num_epochs times before generating
    # an OutOfRange error. If not specified, string_input_producer can cycle
    # through the strings in string_tensor an unlimited number of times.
    #-shuffle: Boolean. If true, the strings are randomly shuffled within each
    # epoch.
    #-seed: An integer (optional). Seed used if shuffle == True.
    #-capacity: An integer. Sets the queue capacity.
    #-shared_name: (optional). If set, this queue will be shared under the given
    # name across multiple sessions. All sessions open to the device which has
    # this queue will be able to access it via the shared_name. Using this in a
    # distributed setting means each name will only be seen by one of the
    # sessions which has access to this operation.
    #-name: A name for the operations (optional).
    #-cancel_op: Cancel op for the queue (optional).
    #Returns:
    #-A queue with the output strings. A QueueRunner for the Queue is added to
    # the current Graph's QUEUE_RUNNER collection.

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    label = tf.decode_raw(features['train/label'], tf.float32)

    # Image content informations

    # BBOX data is actually dense convert it to dense tensor
    nb_obj = tf.cast(features['train/objects_number'], tf.int64)
    bboxes_shape = tf.parallel_stack([nb_obj, 4])
    bboxes = tf.sparse_tensor_to_dense(features['train/bboxes'], default_value=0)
    bboxes = tf.reshape(bboxes, bboxes_shape)
    bboxes_labels = tf.cast(features['train/bb_labels'], tf.int64)

    # get info about images
    width = tf.cast(features['train/width'], tf.int32)
    height = tf.cast(features['train/height'], tf.int32)
    channels = tf.cast(features['train/channels'], tf.int32)

    # get some more info if needed
    comment = tf.cast(features['train/info/comment'], tf.string)
    quality = tf.cast(features['train/info/quality'], tf.float32)
    #Printing this here causes the program to hang
    #print('Comment was: {} and quality was {}'.format(comment.eval(),
    #                                                  quality.eval()))

    # Reshape image data into the original shape
    # aShape must be defined otherwise, get
    # ValueError: All shapes must be fully defined:
    #image = tf.reshape(image, [height, width, channels])
    #One can reshape dynamically an image in tf, but the result cannot be fed
    # into a batch generator, hence the problem !
    # see https://github.com/tensorflow/tensorflow/issues/2604
    image = tf.reshape(image, [768, 1024, 3])
    label = tf.reshape(label, [768, 1024])


    # reformat data if needed.
    
    # Creates batches by randomly shuffling tensors
    batch_size = 2
    example = [image, label, height, width, channels, comment, quality]
    shuffled_batch = tf.train.shuffle_batch(example,
                                            batch_size=batch_size,
                                            capacity=nb_iter*batch_size,
                                            num_threads=1,
                                            min_after_dequeue=4*batch_size,
                                            seed=401,
                                            allow_smaller_final_batch=False)

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
    for batch_index in range(nb_iter):
        img, lbl, h, w, chan, com, qual = sess.run(shuffled_batch)
        print('starting a new batch')
        for i in range(batch_size):
            img = img.astype(np.uint8)
            #plt.subplot(111)
            #plt.imshow(img[i, ...])
            #plt.title('image from tfrecord')
            #plt.show()
            print('Comment was: {} and quality was {}'.format(
                com[i].decode(), qual[i]))

    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
