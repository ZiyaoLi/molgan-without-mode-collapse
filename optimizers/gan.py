import tensorflow as tf
from models.gan import GraphGANModel


class GraphGANOptimizer(object):

    def __init__(self,
                 model: GraphGANModel,
                 learning_rate=1e-3,
                 feature_matching=True):

        self.la = tf.placeholder_with_default(1., shape=())

        with tf.name_scope('losses'):
            eps = tf.random_uniform(tf.shape(model.logits_real)[:1], dtype=model.logits_real.dtype)
            eps4d = tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1)
            eps3d = tf.expand_dims(tf.expand_dims(eps, -1), -1)

            x_int0 = model.adjacency_tensor * eps4d + model.edges_softmax * (1 - eps4d)
            x_int1 = model.node_tensor * eps3d + model.nodes_softmax * (1 - eps3d)

            grad0, grad1 = tf.gradients(
                model.D_x((x_int0, None, x_int1), model.discriminator_units), (x_int0, x_int1))

            self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + \
                                tf.reduce_mean(((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keep_dims=True)

            self.loss_D = - model.logits_real + model.logits_fake  # bi-entropy
            self.loss_G = - model.logits_fake                      # log likelihoods of samples
            self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + \
                          (model.value_logits_fake - model.rewardF) ** 2
            self.loss_RL = - model.value_logits_fake
            self.loss_F = (tf.reduce_mean(model.features_real, 0) - tf.reduce_mean(model.features_fake, 0)) ** 2

        self.loss_D = tf.reduce_mean(self.loss_D)
        self.loss_G = tf.reduce_sum(self.loss_F) if feature_matching else tf.reduce_mean(self.loss_G)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        alpha = tf.abs(tf.stop_gradient(self.loss_G / self.loss_RL))
        self.grad_penalty = tf.reduce_mean(self.grad_penalty)

        with tf.name_scope('train_step'):
            # step for discriminator
            self.train_step_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_D + 10 * self.grad_penalty,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

            # step for generator
            self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_G, lambda: 0.) +
                     tf.cond(tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

            # step for RL reward network
            self.train_step_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_V,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))

#
# class PacGANOptimizer(object):
#
#     def __init__(self,
#                  model: GraphGANModel,
#                  learning_rate=1e-3,
#                  feature_matching=True):
#
#         self.la = tf.placeholder_with_default(1., shape=())
#
#         with tf.name_scope('losses'):
#             eps = tf.random_uniform(tf.shape(model.logits_real)[:1], dtype=model.logits_real.dtype)
#             eps4d = tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1)
#             eps3d = tf.expand_dims(tf.expand_dims(eps, -1), -1)
#
#             x_int0 = model.adjacency_tensor * eps4d + model.edges_softmax * (1 - eps4d)
#             x_int1 = model.node_tensor * eps3d + model.nodes_softmax * (1 - eps3d)
#
#             grad0, grad1 = tf.gradients(
#                 model.D_x((x_int0, None, x_int1), model.discriminator_units), (x_int0, x_int1))
#
#             self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + \
#                                 tf.reduce_mean(((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keep_dims=True)
#
#             self.loss_D = - model.logits_real + model.logits_fake  # bi-entropy
#             self.loss_G = - model.logits_fake                      # log likelihoods of samples
#             self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + \
#                           (model.value_logits_fake - model.rewardF) ** 2
#             self.loss_RL = - model.value_logits_fake
#             self.loss_F = (tf.reduce_mean(model.features_real, 0) - tf.reduce_mean(model.features_fake, 0)) ** 2
#
#         self.loss_D = tf.reduce_mean(self.loss_D)
#         self.loss_G = tf.reduce_sum(self.loss_F) if feature_matching else tf.reduce_mean(self.loss_G)
#         self.loss_V = tf.reduce_mean(self.loss_V)
#         self.loss_RL = tf.reduce_mean(self.loss_RL)
#         alpha = tf.abs(tf.stop_gradient(self.loss_G / self.loss_RL))
#         self.grad_penalty = tf.reduce_mean(self.grad_penalty)
#
#         with tf.name_scope('train_step'):
#             # step for discriminator
#             self.train_step_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
#                 loss=self.loss_D + 10 * self.grad_penalty,
#                 var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
#
#             # step for generator
#             self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
#                 loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_G, lambda: 0.) +
#                      tf.cond(tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
#                 var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
#
#             # step for RL reward network
#             self.train_step_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
#                 loss=self.loss_V,
#                 var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))
