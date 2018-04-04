import tensorflow as tf
import numpy as np

data = np.arange(1, 9 + 1)
data_input = tf.constant(data)
batch_size = 10
nb_iter = 11

batch_shuffle = tf.train.shuffle_batch([data_input],
                                       enqueue_many=True, 
                                       batch_size=batch_size,
                                       capacity=100,
                                       min_after_dequeue=4*batch_size,
                                       num_threads=4,
                                       seed=401,
                                       allow_smaller_final_batch=True)
#allow_smaller_final_batch: if True, a smaller batch value than batch_size
#is returned when the queue is closed and there are not enough elements to fill
#the batch
#capacity: An integer. The maximum number of elements in the queue
#min_after_dequeue: Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
#num_threads: The number of threads enqueuing tensor_list.
#seed: Seed for the random shuffling within the queue.


# the train shuffle batch generates its own ops for parallel computations:
#This function adds the following to the current Graph:
#A shuffling queue into which tensors from tensors are enqueued.
#A dequeue_many operation to create batches from the queue.
#A QueueRunner to QUEUE_RUNNER collection, to enqueue the tensors from tensors.

batch_no_shuffle = tf.train.batch([data_input],
                                  enqueue_many=True,
                                  batch_size=batch_size,
                                  capacity=100,
                                  allow_smaller_final_batch=True)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(nb_iter):
        print(i, sess.run([batch_shuffle, batch_no_shuffle]))
    coord.request_stop()
    coord.join(threads)

#0 [array([75, 36, 42, 37, 34, 63, 14, 89,  9, 24]),
#  array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])]

#1 [array([79, 99, 47, 94, 12, 88, 10, 18, 22, 11]),
#array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])]

#2 [array([35, 13, 64, 16, 45, 54, 12, 66, 33, 57]),
#array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30])]

#3 [array([91, 21, 78, 30,  5, 50, 27, 56,  8, 19]),
#array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40])]

#4 [array([ 6, 15,  3, 76, 77, 21,  2, 31, 49,  4]),
# array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50])]

#5 [array([61, 14, 19, 53,  7, 96, 43, 38, 98, 84]),
# array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60])]

#6 [array([60,  2, 51, 97, 23, 92, 13, 17, 48, 36]),
# array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70])]

#7 [array([29, 17, 70,  5, 85, 83, 28, 44, 61, 81]),
# array([71, 72, 73, 74, 75, 76, 77, 78, 79, 80])]

#8 [array([53, 63,  6, 55, 59, 74, 64, 15, 44, 54]),
# array([81, 82, 83, 84, 85, 86, 87, 88, 89, 90])]

#9 [array([60, 69, 46, 26, 89, 50, 71, 73,  8, 33]),
# array([ 91,  92,  93,  94,  95,  96,  97,  98,  99, 100])]

