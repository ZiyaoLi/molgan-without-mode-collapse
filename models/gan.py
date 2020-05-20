import numpy as np
import tensorflow as tf

from models import postprocess_logits
from utils.layers import multi_dense_layers


class GraphGANModel(object):

    def __init__(self,
                 vertexes,             # max atom num?
                 edges,                # bond num types
                 nodes,                # atom num types
                 embedding_dim,        # z space dim
                 decoder_units,        # tuple, decoder setup (z = Dense(z, dim=units_k)^{(k)})
                 discriminator_units,  # tuple, discr. setup (GCN units, Readout units, MLP units)
                 decoder,              # callable fn. in models.__init__
                 discriminator,        # callable fn. in models.__init__
                 soft_gumbel_softmax=False,
                 hard_gumbel_softmax=False,
                 batch_discriminator=True):

        self.vertexes, self.edges, self.nodes, \
        self.embedding_dim, self.decoder_units, self.discriminator_units, \
        self.decoder, self.discriminator, self.batch_discriminator = \
            vertexes, edges, nodes, embedding_dim, decoder_units, \
            discriminator_units, decoder, discriminator, batch_discriminator

        self.training = tf.placeholder_with_default(False, shape=())
        self.dropout_rate = tf.placeholder_with_default(0., shape=())
        self.soft_gumbel_softmax = tf.placeholder_with_default(soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.placeholder_with_default(hard_gumbel_softmax, shape=())
        self.temperature = tf.placeholder_with_default(1., shape=())   # temperature for softmax

        self.edges_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes, vertexes))
        self.nodes_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes))
        self.embeddings = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim))

        self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.adjacency_tensor = tf.one_hot(self.edges_labels, depth=edges, dtype=tf.float32)
        self.node_tensor = tf.one_hot(self.nodes_labels, depth=nodes, dtype=tf.float32)

        with tf.variable_scope('generator'):
            self.edges_logits, self.nodes_logits = \
                self.decoder(self.embeddings, decoder_units, vertexes, edges, nodes,
                             training=self.training, dropout_rate=self.dropout_rate)

        with tf.name_scope('outputs'):
            (self.edges_softmax, self.nodes_softmax), \
            (self.edges_argmax, self.nodes_argmax), \
            (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
            (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
            (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = \
                postprocess_logits((self.edges_logits, self.nodes_logits), temperature=self.temperature)

            self.edges_hat = tf.case({
                self.soft_gumbel_softmax: lambda: self.edges_gumbel_softmax,
                self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                    self.edges_gumbel_argmax - self.edges_gumbel_softmax) + self.edges_gumbel_softmax},
                default=lambda: self.edges_softmax,
                exclusive=True)

            self.nodes_hat = tf.case({
                self.soft_gumbel_softmax: lambda: self.nodes_gumbel_softmax,
                self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                    self.nodes_gumbel_argmax - self.nodes_gumbel_softmax) + self.nodes_gumbel_softmax},
                default=lambda: self.nodes_softmax,
                exclusive=True)

        with tf.name_scope('D_x_real'):
            self.logits_real, self.features_real = self.D_x(
                (self.adjacency_tensor, None, self.node_tensor), units=discriminator_units)
        with tf.name_scope('D_x_fake'):
            self.logits_fake, self.features_fake = self.D_x(
                (self.edges_hat, None, self.nodes_hat), units=discriminator_units)

        with tf.name_scope('V_x_real'):
            self.value_logits_real = self.V_x(
                (self.adjacency_tensor, None, self.node_tensor), units=discriminator_units)
        with tf.name_scope('V_x_fake'):
            self.value_logits_fake = self.V_x(
                (self.edges_hat, None, self.nodes_hat), units=discriminator_units)

    def D_x(self, inputs, units):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            graph_readouts = self.discriminator(  # units: (GCN units, readout unit)
                inputs, units=units[:-1], training=self.training, dropout_rate=self.dropout_rate)

            mlp_processed = multi_dense_layers(
                graph_readouts, units=units[-1], activation=tf.nn.tanh,
                training=self.training, dropout_rate=self.dropout_rate)

            if self.batch_discriminator:
                outputs_batch = tf.layers.dense(graph_readouts, units[-2] // 8, activation=tf.tanh)
                outputs_batch = tf.layers.dense(tf.reduce_mean(outputs_batch, 0, keep_dims=True), units[-2] // 8,
                                                activation=tf.nn.tanh)
                outputs_batch = tf.tile(outputs_batch, (tf.shape(graph_readouts)[0], 1))

                mlp_processed = tf.concat((mlp_processed, outputs_batch), -1)

            logits = tf.layers.dense(mlp_processed, units=1)

        return logits, mlp_processed

    def V_x(self, inputs, units):  # output reward estimations
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            graph_readouts = self.discriminator(  # units: (GCN units, readout unit)
                inputs, units=units[:-1], training=self.training, dropout_rate=self.dropout_rate)

            graph_readouts = multi_dense_layers(
                graph_readouts, units=units[-1], activation=tf.nn.tanh,
                training=self.training, dropout_rate=self.dropout_rate)

            graph_readouts = tf.layers.dense(graph_readouts, units=1, activation=tf.nn.sigmoid)

        return graph_readouts

    def sample_z(self, batch_dim, mean=0, std=1):
        return np.random.normal(mean, std, size=(batch_dim, self.embedding_dim))
