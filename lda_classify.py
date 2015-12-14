"""
Author : Soubhik Barari

This is an academic project completed for the course COMP 136 : Statistical Pattern Recognition
at Tufts University.

Evaluation of dimensionality reduction by LDA to support 
doc classification using SVM. (Task 2)

"""

import random
import csv

import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", family="serif")

# libsvm import from local directory 
# (comment out following two lines if libsvm installed at root)
import sys
sys.path.append("libsvm/python")

from svmutil import *

from data import *
from model import *


def stddev(lst):
	""" Calculate std dev of a list of numerical values.
	"""
	mn = sum(lst)/float(len(lst))
	return sum((x-mn)**2 for x in lst)


n_trials 	= 30
costs 		= [1,10,100]

f, axes 	= plt.subplots(len(costs))

# get features for documents as bag of words
bow_docs = []
for doc in data100.values():
	word_count = dict(zip(data100v, [0]*len(data100v)))
	for word in doc:
		word_count[word] += 1

	bow_docs.append(word_count.values())

print "** generated `bag-of-words' document feature vectors"



# get features for documents as topic frequencies
tf_docs = []
with open("data100docs.csv", "r") as f:
	line = f.readline()
	while line:
		tf = map(lambda x: int(x), line.split(","))

		tf_docs.append(tf)
		line = f.readline()

print "** generated topic frequency vectors"


# create learning curve for SVMs with different costs (1,10,100)
for i,cost in enumerate(costs):

	print "** started doc classification [SVM cost %i]" % cost

	axes[i].set_title("cost: %i" % cost)
	axes[i].set_ylabel("error")
	axes[i].set_xlabel(r"training set size (% of total set)")

	train_size_errors = {}

	for (color, name, docs) in zip(["g", "b"], ["bag-of-words", "topic frequency"], [bow_docs, tf_docs]):

		print "**\tSVM prediction using `%s' representation" % name

		for trial in range(n_trials):
			if trial != 0 and 10 % trial == 0:
				print "**\t\ttrial %i" % trial

				data = zip(docs, data100class.values())
				random.shuffle(data)
				test_set 	= data[:40]
				train_set 	= data[40:]

				train_sizes = range(10,len(train_set))

				for train_size in train_sizes:
					# training split
					trn_lbls = list(zip(*train_set[:train_size])[1])
					trn_data = list(zip(*train_set[:train_size])[0])

					# SVM cost
					prob 	= svm_problem(trn_lbls, trn_data)
					param 	= svm_parameter("-t 0 -c %i -q" % cost)

					# fit SVM model to training set
					mdl 	= svm_train(prob, param)

					# get SVM predictions on testing set
					tst_lbls = list(zip(*test_set)[1])
					tst_data = list(zip(*test_set)[0])

					(p_labels, p_acc, p_val) = svm_predict(tst_lbls, tst_data, mdl, "-q")

					# report error
					error = (1.0 - float(p_acc[0])/float(100))
					try:
						train_size_errors[train_size].append(error)
					except:
						train_size_errors[train_size] = [error]


		mean_errors 	= map(lambda x: sum(x)/float(len(x)), train_size_errors.values())
		std_dev_errors 	= map(stddev, train_size_errors.values())
		axes[i].errorbar(train_sizes, mean_errors, yerr=std_dev_errors, fmt='-o', label=name, c=color)

	print "** finished doc classification [SVM cost %i]" % cost


plt.legend()
plt.tight_layout()
plt.show()






