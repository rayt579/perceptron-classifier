"""
Implementation of Perceptron algorithm on MNIST dataset. 
"""

import numpy as np


def dot_prod(v1,v2):
	'''
	Returns projection of v1 on v2. 
	'''
	assert v1.size == v2.size, "vectors of different sizes"
	
	# Return scalar value of the dot product
	n = v1.size 
	dp = 0
	for i in xrange(n): 
		dp = dp + (v1[i] * v2[i])
	return dp  

def perceptron(data, labels, n_passes):
	'''
	Returns data matrix of weight vectors, where the last column is 
	the respective counts used for voted/averaged classification.  
	'''
	# Initialization step 
	w = np.zeros(data[0].size)
	m = 0
	c = 1
	n = labels.size
	w_vectors = [w]
	c_vectors = [1]

	# Iteration step. 
	for i in xrange(n):
		if (labels[i] * dot_prod(w,data[i])) <= 0:
			w = w + (labels[i] * data[i])
			w_vectors.append(w)
			c_vectors.append(c)
			m += 1 
		else: 
			c_vectors[m] += 1

	weight_mat = np.insert(w_vectors,(data[i].size), c_vectors, axis=1)
	return weight_mat

def percep_orig_clf(weight_mat, data): 
	'''
	Returns prediction vector for input data, containing either 1 or -1. 
	'''
	# Take the last row from the weight matrix
	x, y = weight_mat.shape
	w = weight_mat[x-1][:-1]

	# Classification rule using dot product
	n, d = data.shape
	pred = []

	for i in xrange(n):
		if np.sign(dot_prod(w,data[i])) == 1:
			pred.append(1)
		else:
			pred.append(-1)
	
	return np.array(pred)

def voted_percep_clf(weight_mat, data):
	'''
	Returns prediction vector using voted classification rule
	'''
	x, y = weight_mat.shape
	weight_vecs = list(weight_mat[:,:-1])
	c_vecs = list(weight_mat[:,y-1])
	
	n, d = data.shape
	pred = []

	# Voted perceptron rule applied to each test point
	for i in xrange(n):
		tot = 0 
		for j in xrange(len(weight_vecs)):
			tot += (c_vecs[j] * np.sign(dot_prod(weight_vecs[j],data[i])))

		# Classification rule for point
		if np.sign(tot) == 1:
			pred.append(1)
		else:
			pred.append(-1)

	return np.array(pred)

def avg_percep_clf(weight_mat, data):
	'''
	Returns prediction vector using averaged classification rule
	'''
	x, y = weight_mat.shape
	weight_vecs = list(weight_mat[:,:-1])
	c_vecs = list(weight_mat[:,y-1])

	n, d = data.shape
	pred = []

	# Get average weight vector
	w = np.zeros(d)
	for j in xrange(len(weight_vecs)):
		w += (c_vecs[j] * weight_vecs[j])

	# Classify the points
	for i in xrange(n):
		if np.sign(dot_prod(w,data[i])) == 1:
			pred.append(1)
		else: 
			pred.append(-1)

	return np.array(pred)

def main():
	# Parse input 
	train = np.genfromtxt("toy_data/toy_train.txt")
	test = np.genfromtxt("toy_data/toy_test.txt")

	n_feat = train[0].size
	# Separate data from labels
	train_labels = train[:,n_feat-1]
	train_data = train[:,:-1]
	test_labels = test[:,n_feat-1]
	test_data = test[:,:-1]


	wm = perceptron(train_data,train_labels,1)
	print avg_percep_clf(wm,test_data)

if __name__ == '__main__':
	main()