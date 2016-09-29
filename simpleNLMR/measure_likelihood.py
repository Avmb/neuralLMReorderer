#! /usr/bin/python

import sys
import cPickle
import numpy as np
from collections import defaultdict
import sklearn.feature_extraction as skfe

from Node import *

from reordererState import ReordererState
from simpleNLMR_1 import SimpleNLMR

def getDepSentenceFeatures(sid, sent, dtype, trDV, taDV):
	nodes, root = createNodes(sent, sid)
	reorderedNodes = sorted(nodes, key = lambda n: n.hint)
	toEmitZId = range(len(reorderedNodes))
	
	trDictL = []
	tasDictL = []
	monosL = []
	vidL = []
	prevState = None
	prevId = 0
	state = ReordererState((nodes, root))
	for j in xrange(len(reorderedNodes) - 1):
		positiveNode = reorderedNodes[j]
		for vid, k in enumerate(toEmitZId):
			candidateNode = nodes[k]
			tasDictL.append(dict(state.targetFeaturesToNode(candidateNode)))
			monosL.append(candidateNode.id == prevId + 1)
			if candidateNode == positiveNode:
				vidL.append(vid)
		
		toEmitZId.remove(positiveNode.id - 1)
		if j < len(reorderedNodes) - 2:
			trDictL.append(dict(state.transitionFeaturesToNode(positiveNode)))
		prevState = state
		state = state.successorToNode(positiveNode)
		prevId = positiveNode.id
	trSeq = trDV.transform(trDictL)
	tasSeq = taDV.transform(tasDictL)
	monosSeq = np.array(monosL, dtype=dtype)
	vidSeq = np.array(vidL, dtype=np.int32)
	return (trSeq, tasSeq, monosSeq, vidSeq)


def testDepSentence(sid, sent, snlmr, trDV, taDV):
	mean_loss =  snlmr.FN_mean_loss_proc_(*getDepSentenceFeatures(sid, sent, snlmr.dtype_, trDV, taDV))
	#loss_seq = snlmr.FN_loss_seq_proc_(*getDepSentenceFeatures(sid, sent, snlmr.dtype_, trDV, taDV))
	#print >> sys.stderr, 'Loss seq:', loss_seq
	#print >> sys.stderr, 'Cumulative loss seq:', np.cumsum(loss_seq)
	return mean_loss

def test(sentF, paramsF, trDV, taDV, monotonic_transition_prob = 0.7, verbose = 0):
	snlmr = SimpleNLMR(make_full_normalized_model = True, random_state = 0, verbose = 3)
	snlmr.fit(transition_features_dim = len(trDV.feature_names_), target_features_dim = len(taDV.feature_names_), monotonic_transition_prob = monotonic_transition_prob)
	snlmr.loadModelParams(paramsF)
	
	loss_acc = 0.0
	n = 0
	for i, sent in  enumerate(readDepSentences(sentF)):
		loss = testDepSentence(i+1, sent, snlmr, trDV, taDV)
		if verbose > 2:
			print >> sys.stderr, "Sentence", i+1, "loss:", loss
		loss_acc += loss
		n += 1
	print loss_acc / n

def main():
	argn = len(sys.argv)
	if (argn != 3):
		usage()
		sys.exit(-1)
	
	dvF = open(sys.argv[1])
	paramsF = open(sys.argv[2])
	tmp = cPickle.load(dvF)
	trDV, taDV = tmp['transitionFeatureDictVectorizer'], tmp['targetFeatureDictVectorizer']
	

	test(sys.stdin, paramsF, trDV, taDV, verbose = 3)

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'dictionary-vectorizers-file parameters-file'

if __name__ == '__main__':
	main()

