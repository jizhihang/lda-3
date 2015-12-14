"""
Author : Soubhik Barari

This is an academic project completed for the course COMP 136 : Statistical Pattern Recognition
at Tufts University.

Evaluation of LDA Gibbs Sampling on various datasets. (Task 1)

"""

import csv

from data import *
from model import *


print "** Artificial LDA"
gibbs_sampler(artificial.values(), K=2, iters=200, log=True)

print "** 20 Newsgroup LDA"
(z, C_t, C_d) = gibbs_sampler(data100.values(), alpha=2.5, beta=0.01, K=20, iters=100, log=True, outfile="newtopicwords.csv")

# save LDA document topics for classification (Task 2)
with open("data100docs.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(C_t)