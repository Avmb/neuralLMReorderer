#! /usr/bin/python

import sys
import numpy as np
from sklearn.utils import check_random_state
from Node import *

def main():
	argn = len(sys.argv)
	if (argn != 4) and (argn != 5):
		usage()
		sys.exit(-1)
	
	outTrainF = open(sys.argv[1], 'w')
	outValidF = open(sys.argv[2], 'w')
	validation_set_size = int(sys.argv[3])
	random_seed = int(sys.argv[4]) if argn == 5 else 0
	rng = check_random_state(random_seed)
	
	print >> sys.stderr, "Loading dataset..."
	sentL = list(readDepSentences(sys.stdin))
	if len(sentL) < validation_set_size:
		print >> sys.stderr, "Error: dataset too small."
		sys.exit(-1)
	
	print >> sys.stderr, "Shuffling..."
	rng.shuffle(sentL)
	
	print >> sys.stderr, "Writing validation set..."
	for sent in sentL[:validation_set_size]:
		for lineL in sent:
			print >> outValidF, '\t'.join(lineL)
		print >> outValidF, ''
	outValidF.close()
	
	print >> sys.stderr, "Writing training set..."
	for sent in sentL[validation_set_size:]:
		for lineL in sent:
			print >> outTrainF, '\t'.join(lineL)
		print >> outTrainF, ''
	outTrainF.close()
	
	print >> sys.stderr, "Done."

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'out-train-file out-validation-file validation-set-size [random-seed]'

if __name__ == '__main__':
	main()



