import tensorflow as tf


def graph_convolution_layer(inputs, units, training, activation=None, dropout_rate=0.):
    '''

    :param inputs: (adjacency_tensor, hidden_tensor, node_tensor).
    adj with shape (b, n, n, c); hidden & node with shape (b, n, d)
    :param units:
    :param training:
    :param activation:
    :param dropout_rate:
    :return:
    '''
    # hidden tensors are always None in the example.
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    adj = tf.transpose(adjacency_tensor[:, :, :, 1:],   # ignore 0-indexed edges.
                       (0, 3, 1, 2))                    # (b, c, n, n) = (?, 4, 9, 9)

    annotations = tf.concat((hidden_tensor, node_tensor), -1) \
        if hidden_tensor is not None else node_tensor

    output = tf.stack([tf.layers.dense(inputs=annotations, units=units)  # diff dense layers for diff bond type
                       for _ in range(adj.shape[1])], 1)                 # (?, 4, 9, 5)

    output = tf.matmul(adj, output)   # (?, 4, 9, units[-1])
    output = tf.reduce_sum(output, 1) + tf.layers.dense(inputs=annotations, units=units)  # res
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def graph_aggregation_layer(inputs, units, training, activation=None, dropout_rate=0.):
    '''
    Readout fn. defined in GG-NN
    :param inputs: (b, n, d)
    :param units: int
    :param training:
    :param activation:
    :param dropout_rate:
    :return:
    '''
    i = tf.layers.dense(inputs, units=units, activation=tf.nn.sigmoid)
    j = tf.layers.dense(inputs, units=units, activation=tf.nn.tanh)
    output = tf.reduce_sum(i * j, 1)    # Eq. 6 in MolGAN, Readout fn in GG-NN
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def multi_dense_layers(inputs, units, training, activation=None, dropout_rate=0.):
    hidden_tensor = inputs
    if isinstance(units, int):
        units = [units]
    for u in units:
        hidden_tensor = tf.layers.dense(hidden_tensor, units=u, activation=activation)
        hidden_tensor = tf.layers.dropout(hidden_tensor, dropout_rate, training=training)

    return hidden_tensor


def multi_graph_convolution_layers(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    for u in units:
        hidden_tensor = graph_convolution_layer(inputs=(adjacency_tensor, hidden_tensor, node_tensor),
                                                units=u, activation=activation, dropout_rate=dropout_rate,
                                                training=training)

    return hidden_tensor
