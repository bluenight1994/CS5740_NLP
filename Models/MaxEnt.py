import numpy as np
import scipy.optimize

class MaxEnt(object):

	def __init__(self, n_class = 2):
		self.n_class = n_class

	def objective_b(self, W, *args):
		X, Y = args[0], args[1]
		lw = 0.
		dw = np.zeros(W.shape[0])
		seg = X.shape[1]
		for i in range(X.shape[0]):
			xi, yi = X[i,:], int(Y[i])
			dw[yi*seg:yi*seg+seg] += xi
			a = np.exp(np.dot(W[yi*seg:yi*seg+seg], xi))
			b = 0
			for j in range(self.n_class):
				b += np.exp(np.dot(W[seg*j:seg*j+seg], xi))
			lw += np.log(float(a) / b)
			for j in range(self.n_class):
				a = np.exp(np.dot(W[j*seg:j*seg+seg],xi))
				dw[j*seg:j*seg+seg] -= (float(a)/b) * xi
		print -lw / (X.shape[0])
		return -lw, -dw


	def objective(self, W, *args):
		X, Y = args[0], args[1]
		lw = 0.
		seg = X.shape[1]
		for i in range(X.shape[0]):
			xi, yi = X[i,:], int(Y[i])
			a = np.exp(np.dot(W[yi*seg:yi*seg+seg], xi))
			b = 0
			for j in range(self.n_class):
				b += np.exp(np.dot(W[seg*j:seg*j+seg], xi))
			lw += np.log(float(a) / b)
		print -lw / (X.shape[0])
		return -lw

	def gradient(self, W, *args):
		X, Y = args[0], args[1]
		dw = np.zeros(W.shape[0])
		for i in range(X.shape[0]):
			xi, yi = X[i,:], int(Y[i])
			seg = X.shape[1]
			dw[yi*seg:yi*seg+seg] += xi
			b = 0
			for j in range(self.n_class):
				b += np.exp(np.dot(W[j*seg:j*seg+seg],xi))
			for j in range(self.n_class):
				a = np.exp(np.dot(W[j*seg:j*seg+seg],xi))
				dw[j*seg:j*seg+seg] -= (float(a)/b) * xi
		return -dw / (X.shape[0])

	def fit(self, X, Y):
		a = X.shape[1]
		self.X = X
		#self.X = np.append(np.ones((X.shape[0],1)), X, axis = 1)
		self.W = np.zeros(self.n_class*a)
		a = scipy.optimize.fmin_l_bfgs_b(self.objective, self.W, fprime=self.gradient,
						 args=(self.X,Y), approx_grad=False, maxfun=100, disp=5)

	def predict(self, X):
		print self.W
		seg = X.shape[1]
		ret = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			tmp = np.zeros(self.n_class)
			for j in range(self.n_class):
				tmp[j] = np.exp(np.dot(X[i],self.W[j*seg:j*seg+seg]))
			print np.argmax(tmp)
			ret[i] = np.argmax(tmp)
		return ret






