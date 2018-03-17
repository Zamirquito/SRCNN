import tensorflow as tf
from utils import data_helper, args
import time
import numpy as np
import os


class SRCNN:
    def __init__(self, flags):
        self.channels = flags.channels

        self.lr = flags.lr
        self.lr_decay = flags.lr_decay
        self.lr_min = flags.lr_min

        self.batch_size = flags.batch_size
        self.epoch = flags.epoch
        self.cheak_freq = flags.cheak_freq
        self.model_dir = flags.model_dir
        self.log_dir = flags.log_dir
        self.sess = tf.Session()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def build_graph(self):
        channels = self.channels
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, channels], name="y")

        with tf.name_scope("conv1"):
            self.w1 = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name="w1")
            self.b1 = tf.Variable(tf.zeros([64]), name='b1')
            with tf.variable_scope("conv1_out"):
                self.conv1 = tf.nn.relu(
                    tf.nn.conv2d(self.x, self.w1, strides=[1, 1, 1, 1], padding="VALID") + self.b1)

            tf.summary.histogram("conv1", self.conv1)
            tf.summary.histogram("w1", self.w1)
            tf.summary.histogram("b1", self.b1)

        with tf.name_scope("conv2"):
            self.w2 = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name="w2")
            self.b2 = tf.Variable(tf.zeros([32]), name='b2')
            with tf.variable_scope("conv2_out"):
                self.conv2 = tf.nn.relu(
                    tf.nn.conv2d(self.conv1, self.w2, strides=[1, 1, 1, 1], padding="VALID") + self.b2)

            tf.summary.histogram("conv2", self.conv2)
            tf.summary.histogram("w2", self.w2)
            tf.summary.histogram("b2", self.b2)

        with tf.name_scope("conv3"):
            self.w3 = tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name="w3")
            self.b3 = tf.Variable(tf.zeros([1]), name='b3')
            with tf.variable_scope("conv3_out"):
                self.conv3 = tf.nn.bias_add(
                    tf.nn.conv2d(self.conv2, self.w3, strides=[1, 1, 1, 1], padding="VALID"), self.b3)

            tf.summary.histogram("conv3", self.conv3)
            tf.summary.histogram("w3", self.w3)
            tf.summary.histogram("b3", self.b3)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.square(self.y - self.conv3))
            tf.summary.scalar("loss", self.loss)

        return self.loss

    def train(self, data, label):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.lr, global_step=global_step, decay_steps=100, decay_rate=0.99)
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
        add_global = global_step.assign_add(1)
        self.sess.run(tf.global_variables_initializer())
        # merged = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        saver = tf.train.Saver()
        batch_size = self.batch_size

        batch_num = len(data) // batch_size
        start_time = time.time()
        for i in range(self.epoch):
            index = np.random.permutation(np.arange(len(data)))
            data = data[index]
            label = label[index]
            for j in range(batch_num):
                batch_data = data[j * batch_size:(j + 1) * batch_size]
                batch_label = label[j * batch_size:(j + 1) * batch_size]
                feed_dict = {self.x: batch_data, self.y: batch_label}
                with tf.device('/gpu:0'):
                    lr_now, step, _, err = self.sess.run([lr, add_global, self.train_op, self.loss], feed_dict=feed_dict)
                    #  lr_now, step, _, err, result = self.sess.run([lr, add_global, self.train_op, self.loss, merged], feed_dict=feed_dict)

                if step % 100 == 0:
                    cur_time = time.time()
                    print("lr: {:.6f}, Epoch: {}, step: {}, loss: {:.5f}, time cost: {:.4f}".format(lr_now, i+1, step, err, cur_time-start_time))

                    start_time = cur_time
                    # summary_writer.add_summary(result, self.step)

            if (i+1) % self.cheak_freq == 0:
                m_name = "model_after_" + str(i) + "_step.ckpt"
                m_name = os.path.join(self.model_dir, m_name)
                saver.save(self.sess, m_name)
        self.sess.close()

    def test(self, data, label):
        feed_dict = {self.x: data, self.y: label}
        with tf.device('/gpu:0'):
            err = self.sess.run([self.loss], feed_dict=feed_dict)
        return err

    def apply(self, image_path):
        pass

