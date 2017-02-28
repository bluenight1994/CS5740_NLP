import tensorflow as tf
import numpy as np

class MLP(object):

	def __init__(self, batch_size = 64, beta = 0.00005, epochs = 50, eta = 0.01, n_class = 2):
		self.weights = None
		self.biases = None
		self.batch_size = batch_size
		self.beta = beta
		self.epochs = epochs
		self.eta = eta
		self.n_class = n_class
		self.display_step = 1

		self.n_hidden_1 = 2048
		self.n_hidden_2 = 2048
		self.n_input = train_x.shape[1]

		x = tf.placeholder(tf.float32, [None, n_input])
		y = tf.placeholder(tf.float32, [None, n_class])

		weights = {
			'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
			'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
			'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
		}

		biases = {
			'b1': tf.Variable(tf.random_normal([n_hidden_1])),
			'b2': tf.Variable(tf.random_normal([n_hidden_2])),
			'out': tf.Variable(tf.random_normal([n_classes]))
        }

        pred = multilayer_perceptron(x, weights, biases)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
							+(tf.nn.l2_loss(weights['h1'])
							+tf.nn.l2_loss(weights['h2'])
							+tf.nn.l2_loss(biases['b1'])
							+tf.nn.l2_loss(biases['b2']))*beta)

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		

	def multilayer_perceptron(self, x, weights, biases):
		# Hidden layer with RELU activation
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.relu(layer_1)
		# Hidden layer with RELU activation
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.relu(layer_2)
		# Output layer with linear activation
		out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
		return out_layer


	def fit(self, train_x, train_y):

        init = tf.initialize_all_variables()

        train_f_y = np.zeros((train_x.shape[0], 5))
        for i, j in enumerate(train_y)
        	train_f_y[i, int(j)] = 1

        with tf.Session() as sess:
        	sess.run(init)
        	for epoch in range(self.epochs):
        		avg_cost = 0.
        		total_batch = int(train_x.shape[0] / batch_size)

        		seq = np.arange(train_x.shape[0])
        		np.random.shuffle(seq)

        		for itr in range(0, total_batch):
        			subseq = seq[itr * batch_size: (itr + 1) * batch_size]

        			_, c = sess.run([optimizer, cost], feed_dict={x: train_x[subseq]
        														y: train_y[subseq]})

        			avg_cost += c / total_batch

        		if epoch % self.display_step == 0:
					print "Epoch:", '%04d' % (epoch+1), "cost=", \
						"{:.9f}".format(avg_cost)

				correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				acc1 = accuracy.eval({x: train_x, y: train_f_y})
				print("Accuracy Train:", acc1)


	def predict(self, test_x):
		res = sess.run(tf.argmax(self.pred, 1), feed_dict={x: test_x})
    	return res


