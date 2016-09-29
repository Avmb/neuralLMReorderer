#! /usr/bin/python

import sys
import cPickle
import numpy as np
from collections import defaultdict
import sklearn.feature_extraction as skfe

from Node import *

from reordererState import ReordererState
from simpleNLMR_2 import SimpleNLMR_2

def getDepSentenceFeatures(sid, sent, dtype, trFM, taFM, reFM):
	nodes, root = createNodes(sent, sid)
	reorderedNodes = sorted(nodes, key = lambda n: n.hint)
	toEmitZId = range(len(reorderedNodes))
	
	trL = []
	tasL = []
	resL = []
	vidL = []
	prevState = None
	prevId = 0
	state = ReordererState((nodes, root))
	for j in xrange(len(reorderedNodes) - 1):
		positiveNode = reorderedNodes[j]
		for vid, k in enumerate(toEmitZId):
			candidateNode = nodes[k]
			tasL.append(state.targetFeaturesToNode(candidateNode))
			resL.append(state.readoutFeaturesToNode(candidateNode))
			if candidateNode == positiveNode:
				vidL.append(vid)
		
		toEmitZId.remove(positiveNode.id - 1)
		if j < len(reorderedNodes) - 2:
			trL.append(state.transitionFeaturesToNode(positiveNode))
		prevState = state
		state = state.successorToNode(positiveNode)
		prevId = positiveNode.id
	trSeq = trFM.transform_seq_to_indices(trL)
	tasSeq = taFM.transform_seq_to_indices(tasL)
	resSeq = reFM.transform_seq_to_sparse(resL)
	vidSeq = np.array(vidL, dtype=np.int32)
	return (trSeq, tasSeq, resSeq, vidSeq)


def testDepSentence(sid, sent, snlmr, trFM, taFM, reFM):
	feats = getDepSentenceFeatures(sid, sent, snlmr.dtype_, trFM, taFM, reFM)
	mean_loss =  snlmr.Mean_loss_proc_(*feats)
	#loss_seq = snlmr.Loss_seq_proc_(*feats)
	#print >> sys.stderr, 'Loss seq:', loss_seq
	#print >> sys.stderr, 'Cumulative loss seq:', np.cumsum(loss_seq)
	return mean_loss

def test(sentF, paramsF, trFM, taFM, reFM, verbose = 0):
	snlmr = SimpleNLMR_2(random_state = 0, verbose = 3)
	snlmr.fit(transition_features_map = trFM, target_features_map = taFM, readout_features_map = reFM, enable_training = False)
	snlmr.loadModelParams(paramsF)
	
	loss_acc = 0.0
	n = 0
	for i, sent in  enumerate(readDepSentences(sentF)):
		loss = testDepSentence(i+1, sent, snlmr, trFM, taFM, reFM)
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
	trFM, taFM, reFM = tmp['transitionFeatureMap'], tmp['targetFeatureMap'], tmp['readoutFeatureMap']
	

	test(sys.stdin, paramsF, trFM, taFM, reFM, verbose = 3)

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'feature-maps-file parameters-file'

if __name__ == '__main__':
	main()

