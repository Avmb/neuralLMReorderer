#! /usr/bin/python

import sys
import cPickle
import numpy as np
from collections import defaultdict
from sortedcontainers.sortedset import SortedSet
import sklearn.feature_extraction as skfe

from Node import *

from reordererState import ReordererState
from featuresBase import prepareDescendents
from simpleNLMR_4 import SimpleNLMR_4

def processDepSentence(sid, sent, snlmr, trFM, taFM, reFM, pfFM, beam_size):
	nodes, root = createNodes(sent, sid)
	prepareDescendents(root)
	state0 = ReordererState((nodes, root), snlmr.getInitialState().reshape((1, -1)))
	beam = SortedSet([(0.0, 0, state0)])
	c = 1
	for j in xrange(len(nodes) - 1):
		nextBeam = SortedSet()
		for totalNegScore, _, state in beam:
			#print >> sys.stderr, totalNegScore
			taL = []
			reL = []
			pfL = []
			pidL = []
			nodeZIdsToEmitL = list(state.nodeZIdsToEmit)
			for candidateNodeZId in nodeZIdsToEmitL:
				candidateNode = nodes[candidateNodeZId]
				taL.append(state.targetFeaturesToNode(candidateNode))
				reL.append(state.readoutFeaturesToNode(candidateNode))
				pfFeats = state.pathFeaturesToNode(candidateNode)
				pfL.extend(pfFeats)
				pidL.append(len(pfFeats))
			
			ta = taFM.transform_seq_to_indices(taL)
			re = reFM.transform_seq_to_sparse(reL)
			pf = pfFM.transform_seq_to_indices(pfL)
			pidSeq = np.array(pidL, dtype=np.int32)
			
			probs = snlmr.Probabilities_with_repeated_state_proc_(state.modelState, len(nodeZIdsToEmitL), ta, re, pf, pidSeq)
			scores = np.log(probs)
			
			for i, candidateNodeZId in enumerate(nodeZIdsToEmitL):
				candidateNode = nodes[candidateNodeZId]
				nextState = state.successorToNode(candidateNode)
				nextBeam.add((totalNegScore - scores[i], c, nextState))
				c += 1
				if len(nextBeam) > beam_size:
					nextBeam.pop()

		if j < len(nodes) - 2:
			trL = []
			modelStateL = []
			for _, _, state in nextBeam:
				node = state.emittedNodesL[-1]
				trL.append(state.prevState.transitionFeaturesToNode(node))
				modelStateL.append(state.prevState.modelState)
			#print >> sys.stderr, trDictL
			tr = trFM.transform_seq_to_indices(trL)
			modelStateMat = np.vstack(modelStateL)
			nextModelStateMat = snlmr.X_state_tp1_proc_(modelStateMat, tr)
			i = 0
			for _, _, state in nextBeam:
				state.modelState = nextModelStateMat[i].reshape((1, -1))
				i += 1
		
		beam = nextBeam	
	bestNegScore, _, bestState = beam.pop(0)
	
	# Emit last node
	lastNodeZId = next(bestState.nodeZIdsToEmit.__iter__())
	lastState = bestState.successorToNode(nodes[lastNodeZId])
	
	# Update hints and output
	lastState.reorderNodes()
	for node in nodes:
		print node.toConllStr()
	print ''
	

def decode(sentF, paramsF, trFM, taFM, reFM, pfFM, state_dim = 100, beam_size = 2, verbose = 0):
	snlmr = SimpleNLMR_4(state_dim = state_dim, random_state = 0, verbose = 3)
	snlmr.fit(transition_features_map = trFM, target_features_map = taFM, readout_features_map = reFM, path_features_map = pfFM, enable_training = False)
	snlmr.loadModelParamsTxt(paramsF)

	for i, sent in enumerate(readDepSentences(sentF)):
		processDepSentence(i+1, sent, snlmr, trFM, taFM, reFM, pfFM, beam_size)
		if verbose > 0:
			print >> sys.stderr, i + 1

def main():
	argn = len(sys.argv)
	if (argn != 3):
		usage()
		sys.exit(-1)
	
	dvF = open(sys.argv[1])
	paramsF = open(sys.argv[2])
	tmp = cPickle.load(dvF)
	trFM, taFM, reFM, pfFM = tmp['transitionFeatureMap'], tmp['targetFeatureMap'], tmp['readoutFeatureMap'], tmp['pathFeatureMap']

	decode(sys.stdin, paramsF, trFM, taFM, reFM, pfFM, state_dim = 100, beam_size = 4, verbose = 1)

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'feature-maps-file parameters-file'

if __name__ == '__main__':
	main()

