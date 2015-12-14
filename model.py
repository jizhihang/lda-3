"""
Author : Soubhik Barari

This is an academic project completed for the course COMP 136 : Statistical Pattern Recognition
at Tufts University.

Collapsed Gibbs sampling implementation for Latent Dirichlet Allocation.

"""


import numpy as np
import random


def np_zeros(shape):
	""" Implementation of numpy's `np.zeros' module.
	"""
	if type(shape) == type(1):
		return [0] * shape

	rows = shape[0]
	cols = shape[1]

	if rows == 1:
		return [0] * cols
	else:
		mat = []
		for i in range(rows):
			mat.append([0]*cols)
	return mat


def gibbs_sampler(docs, K=2, alpha=0.1, beta=0.01, iters=100, 
										log=True, outfile="topicwords.csv"):
	"""
		docs	: list of documents
		K 		: number of topics
		alpha	: dirichlet parameter for topic distribution
		beta 	: dirichlet parameter for word distribution
		iters 	: number of iterations to run sampler
	"""
	if alpha == None:
		alpha = float(50)/K

	# indexes
	w 						= reduce(lambda x,y: x+y, docs)						# word index to actual word
	z 						= [random.randrange(0,K) for word in w] 			# word index to topic
	v 						= [list(set(w)).index(word) for word in w]			# word index to vocab
	d 						= [] 												# word index to doc
	vocab_index_to_word		= list(set(w))											# vocab index to word
	
	for i,doc in enumerate(docs):
		for word in doc:
			d.append(i)

	# constants
	N 						= len(w)											# number of words
	D 						= len(docs)											# number of docs
	V						= len(set(w))										# number of vocab words

	# counts
	doc_to_topic_counts		= np_zeros(shape=(D, K))
	topic_to_word_counts	= np_zeros(shape=(K, V))


	for n in range(N): 															# initialize counts
		topic 	= z[n]
		doc 	= d[n]
		doc_to_topic_counts[doc][topic] 		+= 1
		topic_to_word_counts[topic][v[n]] 		+= 1

	P 						= np_zeros(shape=K)									# topic probabilities

	# permutation
	p 						= range(N)
	random.shuffle(p)

	for _iter in range(iters):
		for n in range(N):

			# pick random element from permutation
			word 			= w[p[n]]
			topic 			= z[p[n]]
			doc 			= d[p[n]]

			doc_to_topic_counts[doc][topic] 		-= 1
			topic_to_word_counts[topic][v[p[n]]] 	-= 1

			# `E step'
			for k in range(K):
				P[k] = (topic_to_word_counts[k][v[p[n]]] + beta)/(V*beta + sum(topic_to_word_counts[k])) * \
					(doc_to_topic_counts[doc][k] + alpha)/(K*alpha + sum(doc_to_topic_counts[doc]))

			# `M step'
			# 	normalize probabilities
			_sum = sum(P)
			for k in range(K):
				P[k] /= _sum

			# 	resample from new topic probabilities
			draw 			= random.uniform(0,1)
			_sub			= 0
			for (k,prob) in sorted(list(enumerate(P)), key=lambda t:t[1]):
				_sub += prob
				if _sub >= draw:
					topic = k
					break
			
			# 	update matrices
			z[p[n]]									= topic
			doc_to_topic_counts[doc][topic] 		+= 1
			topic_to_word_counts[topic][v[p[n]]] 	+= 1


	# results
	f = open(outfile, "w") if outfile else None

	for t,topic in enumerate(topic_to_word_counts):
		print "**\t Topic %i" % (t+1) if log is True else None

		words_by_frequency = sorted(list(enumerate(topic)), key=lambda x: x[1], reverse=True)

		for i,(vocab_index,count) in enumerate(words_by_frequency):
			if i == 5:
				if f:
		 			f.write("\n")
				break

		 	vocab_word = vocab_index_to_word[vocab_index]

		 	# write to file
		 	if count != 0:
		 		f.write(vocab_word+",")

		 	# log using `print'
		 	if count != 0:
		 		print "\t\t%s (%i)" % (vocab_word, count) if log is True else None



	return (z, doc_to_topic_counts, topic_to_word_counts)







