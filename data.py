"""
Author : Soubhik Barari

This is an academic project completed for the course COMP 136 : Statistical Pattern Recognition
at Tufts University.

Datasets for evaluation.

"""

artificial 	= {}
for i in range(1,3):
	with open("artificial/%i" % i, "r") as f:
		artificial[i] = f.read().split()
print "** read in `artifical' dataset"


data100 	= {}
data100v	= []
for i in range(1,101):
	with open("data100/%i" % i, "r") as f:
		data100[i] 	= f.read().split()
		data100v	+= data100[i]

data100v = list(set(data100v))

print "** read in `data100' dataset"


data100class = {}
with open("data100/index.csv", "r") as f:
	
	line = f.readline()
	while line:
		doc 	= int(line.split(",")[0])
		label 	= int(line.split(",")[1])
		data100class[doc] = label
		line 	= f.readline()
print "** read in `data100' class labels"
