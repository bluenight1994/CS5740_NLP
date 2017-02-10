"""
	A multi-class version of Perceptron Classification Model 

	default: n_class = 2 (0, 1)
			 eta     = 0.01
			 epochs  = 50

	if y == predict: pass
	else: w = w - x
"""


import numpy as np

class Perceptron(object):

	def __init__(self, eta = 0.01, epochs = 50, n_class = 2):
		self.eta = eta
		self.epochs = epochs
		self.n_class = n_class


	def train(self, X, Y):
		self.X = np.append(np.ones((X.shape[0],1)), X, axis = 1)
		self.W = [np.zeros(1 + X.shape[1]) for _ in xrange(self.n_class)]

		for _ in range(self.epochs):
			converged = True
			for xi, yi in zip(self.X, Y):
				vec = np.zeros(self.n_class)
				for i in range(self.n_class):
					vec[i] = np.dot(xi, self.W[i])
				y_p = np.argmax(vec)
				if yi == y_p: 
					continue
				else: converged = False
				for i in range(self.n_class):
					if yi == i:
						self.W[i] += self.eta * xi
					else:
						self.W[i] -= self.eta * xi
			if converged: break


	def predict(self, X):
		y = np.zeros(X.shape[0])
		k = 0
		X_t = np.append(np.ones((X.shape[0],1)), X, axis = 1)
		for x in X_t:
			tmp = np.zeros(self.n_class)
			for i in range(self.n_class):
				tmp[i] = np.dot(x, self.W[i])
			y_p = np.argmax(tmp)
			y[k] = y_p
			k += 1
		return y
















