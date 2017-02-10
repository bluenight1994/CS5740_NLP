import numpy as np
import scipy.optimize

class MaxEnt(object):

	def __init__(self, n_class = 2):
		self.n_class = n_class

	def objective(self, W, *args):
		X, Y = args[0], args[1]
		lw = 0
		seg = X.shape[1]
		print seg
		for i in range(X.shape[0]):
			xi, yi = X[i,:], Y[i]
			a = np.exp(np.dot(W[yi*seg:yi*seg+seg], xi))
			b = 0
			for j in range(self.n_class):
				b += np.exp(np.dot(W[seg*j:seg*j+seg], xi))
			lw += np.log(float(a) / b)
		print -lw
		return -lw

	def gradient(self, W, *args):
		X, Y = args[0], args[1]
		dw = np.zeros(W.shape[0])
		for i in range(X.shape[0]):
			xi, yi = X[i,:], Y[i]
			seg = X.shape[1]
			dw[yi*seg:yi*seg+seg] += xi
			b = 0
			for j in range(self.n_class):
				b += np.exp(np.dot(W[j*seg:j*seg+seg],xi))
			for j in range(self.n_class):
				a = np.exp(np.dot(W[j*seg:j*seg+seg],xi))
				dw[j*seg:j*seg+seg] -= (float(a)/b) * xi
		print dw
		return -dw

	def fit(self, X, Y):
		a = X.shape[1]
		self.W = np.random.rand(self.n_class*a)
		a = scipy.optimize.fmin_l_bfgs_b(self.objective, self.W, fprime=self.gradient,
						 args=(X,Y))
		print a[0]
		self.W = a[0]

	def predict(self, X):
		seg = X.shape[1]
		ret = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			tmp = np.zeros(self.n_class)
			for j in range(self.n_class):
				tmp[j] = np.exp(np.dot(X[i],self.W[j*seg:j*seg+seg]))
			ret[i] = np.argmax(tmp)
		return ret




