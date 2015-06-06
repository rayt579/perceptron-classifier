"""
Implementation of Perceptron algorithm on MNIST dataset. 
Here we use binary classification, distinguishing digits '6' from '0'
with a linear classifier with averaged and voting decision rules
"""

import numpy as np


def dot_prod(v1,v2):
	'''
	Returns dot product between v1 and v2
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
	for x in xrange(n_passes):
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

def calc_error(true_labels, pred_labels):
	'''
	Returns percent error of predicted labels
	'''
	assert true_labels.size == pred_labels.size
	n = true_labels.size

	# Calculate the error
	mis_count = 0.0
	for i in xrange(n):
		if pred_labels[i] != true_labels[i]:
			mis_count += 1.0
	return (mis_count/n)

def convert_labels(labels):
	'''
	Returns binary labels, where 0 is '1' and 6 is '-1'
	'''
	new_labels = []
	for point in labels: 
		if point == 0.0:
			new_labels.append(1.0)
		else: 
			new_labels.append(-1.0)
	return np.array(new_labels)

def main():
	'''
	Gets the test and training error of perceptron, voted perceptron, and 
	averaged perceptron at 1, 2, and 3 passes.
	'''
	# Parse input 
	train = np.genfromtxt("data/hw4atrain.txt")
	test = np.genfromtxt("data/hw4atest.txt")

	# Separate data from labels
	n_feat = train[0].size
	train_labels = train[:,n_feat-1]
	train_data = train[:,:-1]
	test_labels = test[:,n_feat-1]
	test_data = test[:,:-1]

	# Convert labels to 1 and -1 for perceptron algorithm
	new_train_labels = convert_labels(train_labels)
	new_test_labels = convert_labels(test_labels)

	# Run 1,2,3 passes of perceptron, write training error
	out = open('summary_output.txt','w')

	# Training Error
	
	for i in xrange(1,4):
		wm = perceptron(train_data,new_train_labels,i)
		percep_labels = percep_orig_clf(wm,train_data)
		voted_percep_labels = voted_percep_clf(wm,train_data)
		avg_percep_labels = avg_percep_clf(wm,train_data)
		out.write('Training Error' + '\n')
		out.write('Pass %s' % i + '\n')
		out.write('Perceptron: %s' % (calc_error(percep_labels,new_train_labels)) + '\n')
		out.write('Voted: %s' % (calc_error(voted_percep_labels,new_train_labels)) + '\n')
		out.write('Average: %s' % (calc_error(avg_percep_labels,new_train_labels)) + '\n')
	
	
	# Test Error
	"""
	for i in xrange(1,4):
		wm = perceptron(train_data,new_train_labels,i)
		percep_labels = percep_orig_clf(wm,test_data)
		voted_percep_labels = voted_percep_clf(wm,test_data)
		avg_percep_labels = avg_percep_clf(wm,test_data)
		out.write('Test Error' + '\n')
		out.write('Pass %s' % i + '\n')
		out.write('Perceptron: %s' % (calc_error(percep_labels,new_test_labels)) + '\n')
		out.write('Voted: %s' % (calc_error(voted_percep_labels,new_test_labels)) + '\n')
		out.write('Average: %s' % (calc_error(avg_percep_labels,new_test_labels)) + '\n')
	"""
if __name__ == '__main__':
	main()