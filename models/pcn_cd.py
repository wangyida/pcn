# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
import tensorflow as tf
from tf_util import mlp, mlp_conv, chamfer, add_train_summary, add_valid_summary
from pc_distance import tf_nndistance

class Model:
    def __init__(self, inputs, gt, alpha):
        self.num_coarse = 1024
        self.grid_size = 4
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs[:,:,0:2])
        self.coarse, self.fine = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.coarse[0], self.fine[0], gt[0]]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder(self, inputs):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = tf.squeeze(mlp_conv(tf.expand_dims(inputs, -2), [128, 256]))
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = tf.squeeze(mlp_conv(tf.expand_dims(features, -2), [512, 1024]))
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [2048, 2048, self.num_coarse * (3 + 12)])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3 + 12])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3 + 12])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)
            # feat_1 = mlp(feat, [512, 512, 3])
            # feat_2 = tf.concat([feat_1, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3 + 12])

        # with tf.variable_scope('folding_1', reuse=tf.AUTO_REUSE):
            fine = mlp(feat, [512, 512, 3 + 12]) + center
        return coarse, fine

    def create_loss(self, coarse, fine, gt, alpha):
        loss_coarse = chamfer(coarse[:,:,0:3], gt[:,:,0:3])
        _, retb, _, retd = tf_nndistance.nn_distance(coarse[:,:,0:3], gt[:,:,0:3])
        for i in range(np.shape(gt)[0]):
            index = tf.expand_dims(retb[i], -1)
            sem_feat = tf.nn.softmax(coarse[i,:,3:], -1)
            sem_gt = tf.cast(tf.one_hot(tf.gather_nd(tf.cast(gt[i,:,3]*80*12, tf.int32), index), 12), tf.float32)
            loss_sem_coarse = tf.reduce_mean(-tf.reduce_sum(
                        0.9 * sem_gt * tf.log(1e-6 + sem_feat) + (1 - 0.9) *
                        (1 - sem_gt) * tf.log(1e-6 + 1 - sem_feat), [1]))
            loss_coarse += loss_sem_coarse
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(fine[:,:,0:3], gt[:,:,0:3])
        _, retb, _, retd = tf_nndistance.nn_distance(fine[:,:,0:3], gt[:,:,0:3])
        for i in range(np.shape(gt)[0]):
            index = tf.expand_dims(retb[i], -1)
            sem_feat = tf.nn.softmax(fine[i,:,3:], -1)
            sem_gt = tf.cast(tf.one_hot(tf.gather_nd(tf.cast(gt[i,:,3]*80*12, tf.int32), index), 12), tf.float32)
            loss_sem_fine = tf.reduce_mean(-tf.reduce_sum(
                        0.9 * sem_gt * tf.log(1e-6 + sem_feat) + (1 - 0.9) *
                        (1 - sem_gt) * tf.log(1e-6 + 1 - sem_feat), [1]))
            loss_fine += loss_sem_fine
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = loss_coarse + alpha * loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]
