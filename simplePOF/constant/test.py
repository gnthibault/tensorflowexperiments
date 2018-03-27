import tensorflow as tf


with tf.name_scope('input'):
    x = tf.constant(value=[10,20,30], name='x')

with tf.name_scope('output'):
    y = tf.Variable(x+1, name='y')

initializer_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer_op)
    print('Final value of y is {}'.format(sess.run(y)))
    print('Another way to compute y is {}'.format(y.eval()))

    # explore collections of variables
    print('One can list the collections of the graph: {}'.format(
        sess.graph.get_all_collection_keys()))
    print('And print the list of variables from a specific collection: {}'.format(
        sess.graph.get_collection('variables')))
    print('One can also get any tensor by its name: {}'.format(
        sess.graph.get_tensor_by_name('input/x:0')))

    # explore ops
    print('More generally, one can get a list of all operations of the graph '
        '{}'.format(sess.graph.get_operations()))
    print('And then fetch an operation by its name {}'.format(
        sess.graph.get_operation_by_name('input/x')))
    op = sess.graph.get_operation_by_name('input/x')
    print('And eventually get inputs: {} outputs: {} or some infos out of an '
        'operation: {}'.format(op.inputs, op.outputs, op.get_attr('dtype')))
