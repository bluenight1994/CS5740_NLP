# code structure reference from google tensorflow tutorial
# add feature:
#   1. sentence boundary --- "EOS" token
#   2. CBOW
#   3. down_sample

import numpy as np
import tensorflow as tf
import re
import collections
import random
import math

class word2vec(object):

    def __init__(self):
        self.frequency_filtering = 100000
        self.data_index = 0
        self.batch_size = 128
        self.embedding_size = 200
        self.window_size = 2
        self.negative_sample = 32
        self.lr = 0.5
        self.min_lr = 0.01

    def read_data(self, filepath):
        words = []
        raw = [line for line in open(filepath)]
        for line in raw:
            sentence = filter(None, re.split(r'[^a-zA-Z\']', line.lower()))
            sentence.append("EOS")
            words.extend(sentence)
        counter = [['UNK', -1]]
        counter.extend(collections.Counter(words).most_common(self.frequency_filtering))

        vocab = {}
        n_unknown = 0
        data = []
        for word, _ in counter:
            vocab[word] = len(vocab)
        for word in words:
            if word in vocab:
                idx = vocab[word]
            else:
                idx = 0
                n_unknown += 1
            data.append(idx)
        counter[0][1] = n_unknown
        vocab_inv = dict(zip(vocab.values(), vocab.keys()))
        self.data = data
        self.vocab = vocab
        self.vocab_inv = vocab_inv
        self.counter = counter


    def generate_cbow_batch(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[target]
                labels[i * num_skips + j, 0] = buffer[skip_window]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels


    def generate_skip_gram_batch(self, batch_size, skip_window):
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        cnt = 0
        while cnt < batch_size:
            if self.vocab_inv[self.data[self.data_index]] == "EOS":
                self.data_index = (self.data_index + 1) % len(self.data)
            for i in range(1, skip_window + 1):
                if cnt == batch_size or self.data_index - i < 0: break
                if self.vocab_inv[self.data[self.data_index - i]] == "EOS":
                    break
                else:
                    batch[cnt] = self.data[self.data_index]
                    labels[cnt] = self.data[self.data_index - i]
                    cnt += 1
            for i in range(1, skip_window + 1):
                if cnt == batch_size: break
                if self.vocab_inv[self.data[self.data_index + i]] == "EOS":
                    break
                else:
                    batch[cnt] = self.data[self.data_index]
                    labels[cnt] = self.data[self.data_index + i]
                    cnt += 1
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels


    def build_model(self):
        self.graph = tf.Graph()
        vocabulary_size = len(self.vocab)
        batch_size = self.batch_size
        embedding_size = self.embedding_size
        negative_sample = self.negative_sample
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            self.learn_rate = tf.placeholder(tf.float32, shape=[])

            with tf.device('/cpu:0'):
                self.embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                self.nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))

                self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=self.train_labels,
                               inputs=self.embed,
                               num_sampled=negative_sample,
                               num_classes=vocabulary_size))

            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / self.norm
            self.initialize = tf.initialize_all_variables()


    def build_graph_cbow(self):
        self.graph = tf.Graph()
        vocabulary_size = len(self.vocab)
        batch_size = self.batch_size
        embedding_size = self.embedding_size
        negative_sample = self.negative_sample
        with self.graph.as_default():
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            with tf.device('/cpu:0'):
                self.embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
                self.reduced_embed = tf.div(tf.reduce_sum(self.embed, 1), self.window_size * 2)

                self.nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))

                self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.reduced_embed, self.train_labels,
                               negative_sample, vocabulary_size))

            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embedding / self.norm
            self.initialize = tf.initialize_all_variables()



    def train(self, epochs):
        self.decay = (self.min_lr - self.lr) / (len(self.data) * self.window_size * 2 * epochs)
        with tf.Session(graph=self.graph) as session:
            self.initialize.run()

            average_loss = 0
            for i in xrange(epochs):
                print "epochs" + str(i + 1)
                self.data_index = 0
                step = 0
                while self.data_index < len(self.data):
                    step += 1
                    batch_inputs, batch_labels = self.generate_skip_gram_batch(
                        self.batch_size, self.window_size)
                    # dynamic update learning rate
                    self.lr = max(self.min_lr, self.lr + self.decay)
                    feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels, self.learn_rate: self.lr}
                    _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    average_loss += loss_val

                    if step % 2000 == 0:
                        if step > 0:
                            average_loss /= 2000
                        print "learn rate: ", self.lr
                        print("Average loss at step ", step, ": ", average_loss)
                        average_loss = 0

            self.final_embedding = self.normalized_embeddings.eval()

def main():
    w2v = word2vec()
    w2v.read_data('training-data.1m')
    w2v.build_model()
    w2v.train(1)

if __name__ == '__main__':
    main()