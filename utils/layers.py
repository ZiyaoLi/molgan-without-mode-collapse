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


def graph_readout_layer(inputs, units, training, activation=None, dropout_rate=0.):
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


def masked_softmax(logits, masks):
    logits -= tf.reduce_max(logits, axis=-1, keep_dims=True)   # (b, nb, n, n)
    energy = tf.exp(logits) * masks
    output = energy / (tf.reduce_sum(energy, axis=-1, keep_dims=True) + 1e-6)
    return output

#
# def flat_masked_softmax(logits, masks):
#     logits -= tf.reduce_max(logits, axis=-1, keepdims=True)  # (b, n, n)
#     energy = tf.exp(logits) * masks
#     output = energy / (tf.reduce_sum(energy, axis=-1, keepdims=True) + 1e-6)
#     return output


def flat_attn_head(query, key, value, flat_adj, units, residual):
    f_q = tf.layers.dense(query, 1)
    f_k = tf.layers.dense(key, 1)
    f_v = tf.layers.dense(value, units)
    logits = f_q + tf.transpose(f_k, (0, 2, 1))
    coefs = masked_softmax(logits, flat_adj)

    outputs = tf.matmul(coefs, f_v)
    if residual:
        outputs += f_v

    return outputs


def attn_head(queries, keys, values, adj, units, residual):
    f_q = tf.stack([tf.layers.dense(queries, 1) for _ in range(adj.shape[1])], 1)     # (b, nb, n, 1)
    f_k = tf.stack([tf.layers.dense(keys, 1) for _ in range(adj.shape[1])], 1)        # (b, nb, n, 1)

    f_v = tf.stack([tf.layers.dense(values, units) for _ in range(adj.shape[1])], 1)  # (b, nb, n, u)

    logits = f_q + tf.transpose(f_k, (0, 1, 3, 2))  # (b, nb, n, n)
    coefs = masked_softmax(logits, adj)             # (b, nb, n, n)

    outputs = tf.matmul(coefs, f_v)                 # (b, nb, n, n)

    if residual:
        outputs += f_v

    return outputs


def gat_layer(inputs, units, n_heads, training, activation=None, dropout_rate=0.):
    '''
    :param inputs: (adjacency_tensor, hidden_tensor, node_tensor).
    adj with shape (b, n, n, c); hidden & node with shape (b, n, d)
    :param units:
    :param n_heads: num of attn heads
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

    output = tf.concat([attn_head(
        annotations, annotations, annotations, adj, units=units // n_heads, residual=False
        ) for _ in range(n_heads)], axis=-1)
    output = tf.reduce_sum(output, 1) + tf.layers.dense(inputs=annotations, units=output.shape[-1])  # res
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def flat_gat_layer(inputs, units, n_heads, training, activation=None, dropout_rate=0.):
    '''
    :param inputs: (adjacency_tensor, hidden_tensor, node_tensor).
    adj with shape (b, n, n, c); hidden & node with shape (b, n, d)
    :param units:
    :param n_heads: num of attn heads
    :param training:
    :param activation:
    :param dropout_rate:
    :return:
    '''
    # hidden tensors are always None in the example.
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    adj = tf.transpose(adjacency_tensor[:, :, :, 1:],   # ignore 0-indexed edges.
                       (0, 3, 1, 2))                    # (b, c, n, n) = (?, 4, 9, 9)
    flat_adj = tf.reduce_sum(adj, axis=1, keepdims=False)

    annotations = tf.concat((hidden_tensor, node_tensor), -1) \
        if hidden_tensor is not None else node_tensor

    output = tf.concat([flat_attn_head(
        annotations, annotations, annotations, flat_adj, units=units // n_heads, residual=False
        ) for _ in range(n_heads)], axis=-1)
    output = output + tf.layers.dense(inputs=annotations, units=output.shape[-1])  # res
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def multi_gat_layers(inputs, units, n_heads, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    for u in units:
        hidden_tensor = gat_layer(inputs=(adjacency_tensor, hidden_tensor, node_tensor),
                                  units=u, n_heads=n_heads, activation=activation, dropout_rate=dropout_rate,
                                  training=training)
    return hidden_tensor


def multi_flat_gat_layers(inputs, units, n_heads, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    for u in units:
        hidden_tensor = flat_gat_layer(inputs=(adjacency_tensor, hidden_tensor, node_tensor),
                                       units=u, n_heads=n_heads, activation=activation, dropout_rate=dropout_rate,
                                       training=training)
    return hidden_tensor

