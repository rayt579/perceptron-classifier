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
	the respective counts.  
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

	return np.insert(w_vectors,(data[i].size), c_vectors, axis=1)


def main():
	# Parse input 
	train = np.genfromtxt("toy_data/toy_test.txt")

	# Separate data from labels
	train_labels = train[:,2]
	train_data = train[:,:-1]

	print perceptron(train_data,train_labels,1)

if __name__ == '__main__':
	main()